"""Per-sample rollout monitor for run_from_scratch.ipynb (also reused by the Section 8 training monitor).

Public entry point:
    watch_rollout(work_dir=None, poll=1.0, maxrows=8)

Builds a per-sample view of the CURRENT rollout from rollout.log + harbor.log +
cc_trials, refreshes in place, and auto-stops when the ray job succeeds or every
sample is submitted (interrupt/stop still works). The training monitor reuses
rollout_panel()/_tail() from here.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

# bracket-optional timestamp: matches train.log (unbracketed) and rollout.log ([..])
TS = re.compile(r"\[?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SUBMIT = "COMPLETE_TASK_AND_SUBMIT"  # mini-swe-agent finish sentinel (skipped in the view)
_CLR = {"green": "\033[32m", "yellow": "\033[33m", "gray": "\033[90m"}


def default_work_dir(work_dir=None):
    return (work_dir or os.environ.get("WORK_DIR")
            or os.path.join(os.environ.get("EX", os.getcwd()), "runtime", "work"))


def _tail(p, n):
    return subprocess.run(["tail", "-n", str(n), p], capture_output=True, text=True).stdout if os.path.exists(p) else ""


def _head(p, n):
    return subprocess.run(["head", "-n", str(n), p], capture_output=True, text=True).stdout if os.path.exists(p) else ""


def _short(s, n=120):
    return " ".join(str(s).split())[:n]


def _state(e):  # green=Submitted, yellow=running, gray=other
    c = _CLR["green"] if e == "Submitted" else _CLR["yellow"] if e == "running" else _CLR["gray"]
    return f"{c}{e}\033[0m"


def _reward(r):  # green when the sample passed (reward>0), else gray
    if r is None:
        return f"{_CLR['gray']}reward=?\033[0m"
    c = _CLR["green"] if r > 0 else _CLR["gray"]
    return f"{c}reward={r}\033[0m"


def rollout_started(log):
    return any(k in _tail(log, 400) for k in ("Rollout generation", "Decode batch", "submitted successfully"))


def run_start(log):  # first timestamp in this run's (truncated) log
    m = TS.search(_head(log, 120))
    return m.group(1) if m else None


def harbor_status(harbor_log, since):  # only lines with timestamp >= since (this run)
    st = {}
    if os.path.exists(harbor_log):
        for ln in open(harbor_log):
            m = TS.match(ln)
            if since and (not m or m.group(1) < since):
                continue
            r = re.search(r"Running instance: (\S+)", ln)
            if r:
                st.setdefault(r.group(1), ("running", None))
            f = re.search(r"Instance (\S+) finished: exit_status=(\w+), reward=([0-9.]+)", ln)
            if f:
                st[f.group(1)] = (f.group(2), float(f.group(3)))
    return st


def trial_info(d):
    # per-SAMPLE truth: (turns, last_real_cmd, its_result, exit_status, reward)
    turns = None
    cmd = res = ""
    exit_status = "running"
    reward = None
    rp = f"{d}/verifier/reward.txt"  # authoritative per-sample reward
    if os.path.exists(rp):
        try:
            reward = float(open(rp).read().strip())
        except Exception:
            pass
    for fn in ("mini-swe-agent.trajectory.json", "trajectory.json"):
        p = f"{d}/agent/{fn}"
        if not os.path.exists(p):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        msgs = j.get("messages", [])
        if not msgs:
            continue
        exit_status = (j.get("info", {}) or {}).get("exit_status", "?")
        turns = sum(m.get("role") == "assistant" for m in msgs)
        for i, m in enumerate(msgs):  # last real (non-submit) command + its result
            if m.get("role") == "assistant":
                acts = (m.get("extra", {}) or {}).get("actions", [])
                c = acts[0].get("command", "") if acts else ""
                if c and SUBMIT not in c:
                    cmd = _short(c)
                    if i + 1 < len(msgs) and msgs[i + 1].get("role") == "tool":
                        res = _short(msgs[i + 1].get("content"))
        break
    return turns, cmd, res, exit_status, reward


def rollout_panel(log, trials_dir, harbor_log, maxrows=8, scope_ep=None, title="this run", include_tail=True):
    """Per-sample rollout view. Returns (lines, done, reason).

    scope_ep:     only count trial dirs modified at/after this epoch (defaults to the run start).
                  The training monitor passes the last finished step's time to scope to the CURRENT step.
    title:        header label, e.g. "this run" or "step 1 rollout".
    include_tail: prepend this log's tail (the training monitor prints its own tail, so it skips this).
    """
    since = run_start(log)
    run_ep = datetime.strptime(since, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() if since else 0
    if scope_ep is None:
        scope_ep = run_ep
    hs = harbor_status(harbor_log, since)
    names = set(hs)
    nrun_h = sum(v[0] == "running" for v in hs.values())  # trials in flight (per harbor)
    cand = [d for d in glob.glob(f"{trials_dir}/code_contests-*__*") if os.path.basename(d).split("__")[0] in names]
    cand.sort(key=os.path.getmtime, reverse=True)
    trials = [d for d in cand if os.path.getmtime(d) >= scope_ep - 5]  # scoped (no fallback -> no leakage)
    infos = [(os.path.basename(d).split("__")[0], *trial_info(d)) for d in trials]
    nprob = len({n for n, *_ in infos})
    npass = sum((r or 0) > 0 for *_, r in infos)

    def _bucket(e):  # group every sample into one of three headings
        if e == "Submitted":
            return "submitted"
        if e in ("running", "?", "", None):
            return "running"  # in-flight / not yet finalised
        return "other"  # LimitsExceeded, FormatError, AgentError, ...

    nsub = sum(_bucket(e) == "submitted" for _, _, _, _, e, _ in infos)
    nrun = sum(_bucket(e) == "running" for _, _, _, _, e, _ in infos)
    noth = len(infos) - nsub - nrun
    out = []
    if include_tail:
        out += [f"=== {os.path.basename(log)} (tail) ===", _tail(log, 10).rstrip(), ""]
    out += [f"=== {title}: {nprob} problems | {len(trials)} samples | submitted {nsub} | "
            f"running {nrun} | other {noth} | reward>0: {npass} ==="]
    rows = list(enumerate(infos if maxrows is None else infos[:maxrows]))  # maxrows=None -> show all

    def _emit(header, keep, show_state):
        sel = [(i, t) for i, t in rows if _bucket(t[4]) == keep]
        out.append(f"-- {header} ({len(sel)}) --")
        for i, (name, turns, cmd, res, exit_status, reward) in sel:
            st = f"  {_state(exit_status)}" if show_state else ""  # show the specific status in "Other"
            out.append(f"[{i}] {name}{st}  {_reward(reward)}  turns={turns}")
            if cmd:
                out.append(f"     LLM's cmd: {cmd}")
            if res:
                out.append(f"     Env out: {res}")

    _emit("Submitted", "submitted", False)
    _emit("running", "running", False)
    _emit("Other", "other", True)
    job_done = "succeeded" in _tail(log, 30)
    batch_done = len(trials) > 0 and nrun_h == 0 and nsub == len(trials)
    return out, (job_done or batch_done), ("job succeeded" if job_done else "all samples submitted")


def watch_rollout(work_dir=None, poll=1.0, maxrows=8, wait_steps=150):
    """Live per-sample rollout view for the CURRENT run; auto-stops on completion. Interrupt to stop."""
    from IPython.display import clear_output

    work = default_work_dir(work_dir)
    rollout_log, harbor_log, trials = f"{work}/rollout.log", f"{work}/harbor.log", f"{work}/cc_trials"
    for _ in range(wait_steps):  # gate: wait up to ~5 min for the run to start
        if rollout_started(rollout_log):
            break
        clear_output(wait=True)
        print("waiting for rollout to start...")
        time.sleep(2)
    try:
        while True:
            lines, done, reason = rollout_panel(rollout_log, trials, harbor_log, maxrows=maxrows)
            clear_output(wait=True)
            print("\n".join(lines))
            if done:
                print(f"\n[done \u2014 {reason}]")
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\n[stopped]")
