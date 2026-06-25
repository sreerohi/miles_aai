#!/usr/bin/env python3
"""Standalone live training monitor for qwen3-codecontests.

Terminal equivalent of Section 8's notebook monitor. Run it in a SEPARATE
shell after launching run_training.sh; it is read-only and only tails logs
under the work dir, so it never affects training.

Usage:
    python3 monitor.py                       # raw live train.log tail (last 5 lines) by default
    WORK_DIR=/path/to/work python3 monitor.py
    python3 monitor.py --work-dir /path/to/work --poll 5
    python3 monitor.py --tail-lines 40       # raw tail, last 40 lines
    python3 monitor.py --curated             # only milestone lines (step/rollout/perf/checkpoint)

Works on the host or inside the miles_swe container: the repo is
identity-mounted, so $EX/runtime/work/train.log resolves to the same file
either way. Interrupt (Ctrl-C) to stop.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types

# --------------------------------------------------------------------------- #
# Path setup: anchor everything to THIS file's location so it works no matter
# what the current working directory is.
# --------------------------------------------------------------------------- #
EX = os.path.dirname(os.path.abspath(__file__))
MON_DIR = os.path.join(EX, "notebook-monitoring")
sys.path.insert(0, MON_DIR)

# default_work_dir() falls back to $EX/runtime/work; make sure $EX is set even
# if this shell didn't export it (the training shell did, but this one may not).
os.environ.setdefault("EX", EX)

# training_monitor imports IPython.display at module load for its Plotly charts,
# which only render in a notebook. We only want its log parsers here, so stub
# IPython.display with no-ops -> the import works in a plain shell (no Jupyter).
if "IPython.display" not in sys.modules:
    _fake_ipy = types.ModuleType("IPython")
    _fake_disp = types.ModuleType("IPython.display")
    _fake_disp.clear_output = lambda *a, **k: None
    _fake_disp.display = lambda *a, **k: None
    _fake_ipy.display = _fake_disp
    sys.modules.setdefault("IPython", _fake_ipy)
    sys.modules["IPython.display"] = _fake_disp

import rollout_monitor  # noqa: E402
import training_monitor  # noqa: E402  (parsers only; chart rendering is stubbed out)


def _clear():
    # ANSI clear-screen + home cursor: terminal stand-in for clear_output().
    print("\033[2J\033[H", end="")


def _log_age(path, stale_after):
    """Human-readable age of `path`'s last write, plus a STALE flag.

    Surfaces the common gotcha where a previous run's train.log is still on
    disk but no live run is writing to it yet (setup/prepare phases truncate
    + write the log only once the training step starts).
    """
    if not os.path.exists(path):
        return "missing"
    age = max(0.0, time.time() - os.path.getmtime(path))
    if age < 90:
        human = f"{age:.0f}s"
    elif age < 5400:
        human = f"{age / 60:.1f}m"
    elif age < 172800:
        human = f"{age / 3600:.1f}h"
    else:
        human = f"{age / 86400:.1f}d"
    flag = "  *** STALE — no live run writing here? ***" if age > stale_after else ""
    return f"last write {human} ago{flag}"


def render_table(s):
    """Text version of training_monitor.render_charts (no Plotly)."""
    xs = sorted(s)
    print(f"{'step':>4} {'step_t':>7} {'roll_t':>7} {'reward':>7} {'trunc':>6}")
    for n in xs:
        d = s[n]
        st, rt = d.get("step_time"), d.get("rollout_time")
        rw, tr = d.get("raw_reward"), d.get("truncated")
        print(
            f"{n:>4} "
            f"{(('%.0f' % st) if st is not None else '-'):>7} "
            f"{(('%.0f' % rt) if rt is not None else '-'):>7} "
            f"{(('%.3f' % rw) if rw is not None else '-'):>7} "
            f"{(('%.3f' % tr) if tr is not None else '-'):>6}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=None, help="defaults to $WORK_DIR, else $EX/runtime/work")
    ap.add_argument("--poll", type=float, default=8.0, help="refresh interval in seconds (default: 8)")
    ap.add_argument("--curated", action="store_true",
                    help="show only curated milestone lines (step/rollout/perf/checkpoint/traceback) "
                         "instead of the default raw live tail")
    ap.add_argument("--tail-lines", type=int, default=None,
                    help="number of train.log lines to show in the tail "
                         "(default: 5 raw, or 12 with --curated)")
    args = ap.parse_args()

    raw = not args.curated
    tail_lines = args.tail_lines if args.tail_lines is not None else (12 if args.curated else 5)

    work = rollout_monitor.default_work_dir(args.work_dir)
    tlog = f"{work}/train.log"
    trials = f"{work}/cc_trials"
    harbor_log = f"{work}/harbor.log"
    print(f"[monitor] work_dir  = {work}")
    print(f"[monitor] train.log = {tlog}")
    if not os.path.exists(tlog):
        print("[monitor] train.log not found yet; waiting for training to start...")

    try:
        while True:
            s = training_monitor.parse_steps(tlog)
            step_ts = sorted(v["ts"] for v in s.values() if "ts" in v)
            cur_step = len(step_ts)  # the rollout step currently in progress
            _clear()

            # (1) Pinned W&B link + log freshness (so a stale leftover log is obvious).
            url = training_monitor.wandb_url(tlog)
            print(f"W&B run: {url}" if url else "W&B run: (link not in train.log yet)")
            print(f"train.log: {_log_age(tlog, stale_after=max(60.0, args.poll * 4))}")

            # (2) train.log tail. Default = the full unfiltered stream (every line,
            # like `tail -f`); --curated = only step/rollout/perf/checkpoint/traceback.
            if raw:
                # Pull exactly tail_lines straight off the file -> a true live stream.
                lines = rollout_monitor._tail(tlog, tail_lines).splitlines()
                loglines = [training_monitor.ANSI.sub("", l) for l in lines]
                header = f"=== train.log (raw tail, last {tail_lines}) ==="
            else:
                # Scan a wider window, keep only step/rollout/perf/checkpoint/traceback lines.
                loglines = [
                    training_monitor.ANSI.sub("", l)
                    for l in rollout_monitor._tail(tlog, 300).splitlines()
                    if training_monitor.LOGPAT.search(l)
                ][-tail_lines:]
                header = "=== train.log (tail) ==="
            print(f"\n{header}")
            print("\n".join(loglines) or "(waiting for log...)")

            # (3) Per-sample rollout panel for the CURRENT step.
            scope_ep = step_ts[-1] if step_ts else None
            plines, _done, _reason = rollout_monitor.rollout_panel(
                tlog, trials, harbor_log, scope_ep=scope_ep, maxrows=None,
                title=f"step {cur_step} rollout", include_tail=False,
            )
            if any(l.startswith("[") for l in plines):  # only once this step has samples
                print()
                print("\n".join(plines))

            # (4) Metrics table once any step data exists (Plotly replaced by text).
            if s:
                print()
                render_table(s)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n[stopped]")


if __name__ == "__main__":
    main()
