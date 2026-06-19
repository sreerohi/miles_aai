# qwen3-codecontests — Qwen3-8B on CodeContests (Miles + Megatron + Harbor)

Self-contained example to RL-train **Qwen3-8B** (dense) on the **CodeContests**
Harbor dataset with Miles + Megatron + Harbor, using the **terminus-2** agent.

Everything lives under `/home/rohith/miles_qwen` only — the sibling
`/home/rohith/miles` checkout is never modified.

This is a standalone copy of the agent-agnostic Miles<->Harbor glue
(`server.py`, `generate.py`, `swe_agent_function.py`) plus Qwen3/CodeContests
tooling, so nothing here depends on the `swe-agent-v2/` example. The agent is a
runtime choice (`agent_name=terminus-2`), not tied to mini-swe-agent.

## Layout

```
qwen3-codecontests/
  run-qwen3-codecontests.py   # launcher (dense Qwen3-1.7B single-GPU; EP=1; qwen25/qwen3 parsers; mini-swe-agent)
  launcher_args.py            # dependency-free train-arg builder (single source of truth)
  harbor/                     # Harbor agent glue + agent configs (importable package)
    server.py                 #   Harbor orchestration server — run in agent_env
    swe_agent_function.py     #   Miles custom agent function (harbor.swe_agent_function.run)
    generate.py               #   reward_func + RolloutFn (harbor.generate.*)
    codecontests.yaml         #   mini-swe-agent config (MSWEA_CONFIG_FILE)
    swe_net_override.yaml     #   Harbor compose override -> task containers join swe-net (HARBOR_EXTRA_DOCKER_COMPOSE)
    terminus_codecontests.yaml#   legacy terminus-2 config (superseded by mini-swe-agent)
  data_prep/                  # dataset preparation (run once, before training)
    extract_codecontests.py   #   HF parquet (path + task_binary tar) -> ~/harbor_tasks_cc (no SkyRL)
    build_cc_jsonl.py         #   task dirs -> cc_train.jsonl (prompt + instance_id + agent_name=mini-swe-agent)
    split_by_difficulty.py    #   cc_train.jsonl -> data/cc_train_{easy,medium,...}.jsonl
  data/                       # generated difficulty-split prompt jsonls
  monitoring/                 # live dashboards / collectors (manual, not part of training loop)
    monitor.sh, cc_train_watch.sh, cc_harbor_watch.sh, status.sh, gpu_sampler.sh, stage_util.py
  tests/                      # pytest gates
```

## Quick start

```bash
cd /home/rohith/miles_qwen
EX=examples/experimental/qwen3-codecontests

# 1. unit tests (no services needed)  [host needs: pytest pyyaml pyarrow]
python3 -m pytest $EX/tests -m "not integration" -q

# 2. dataset -> Harbor task dirs (no SkyRL)
python3 $EX/data_prep/extract_codecontests.py --dataset open-thoughts/CodeContests --out ~/harbor_tasks_cc

# 3. prompt jsonl (agent_name=mini-swe-agent) + difficulty splits
python3 $EX/data_prep/build_cc_jsonl.py --tasks ~/harbor_tasks_cc --out ~/cc_train.jsonl
python3 $EX/data_prep/split_by_difficulty.py --tasks ~/harbor_tasks_cc --out-dir $EX/data

# 4. live monitor (own terminal)
bash $EX/monitoring/monitor.sh
```

## Harbor server (run inside agent_env) — note the harbor/ paths

```bash
EX=/root/miles_qwen/examples/experimental/qwen3-codecontests
export OPENAI_API_KEY=dummy MSWEA_API_KEY=dummy
export MSWEA_CONFIG_FILE=$EX/harbor/codecontests.yaml
export HARBOR_EXTRA_DOCKER_COMPOSE=$EX/harbor/swe_net_override.yaml
export HARBOR_DELETE_CONTAINERS=true HARBOR_TASKS_DIR=/root/harbor_tasks_cc
export HARBOR_TRIALS_DIR=/mnt/nvme4n1p1/rohith/work/cc_trials
cd $EX && PYTHONPATH=/root/miles_qwen python3 harbor/server.py --port 11000 --max-concurrent 64
```

The launcher must stay in this directory: it resolves
`FULLY_ASYNC_DIR = SCRIPT_DIR.parent.parent/"fully_async"` and puts `SCRIPT_DIR`
on `PYTHONPATH`, where the copied `swe_agent_function.py` / `generate.py` are
found via `--custom-agent-function-path` / `--custom-rm-path`.

See the plan for the full step-by-step and which container patches to keep/drop.

## Status (2026-06-10)

Pipeline is stood up and validated end-to-end EXCEPT the final reward step:

- Containers `miles_swe` + `agent_env` launched mounting `miles_qwen` + data
  (old GLM ones preserved as `*_glm`). Patches: `postproc_lock` applied, TITO
  present, MoE/8h absent, MLA skipped (gate passes).
- CodeContests extracted (9644 tasks), jsonls built, Qwen3-8B converted to
  `torch_dist` (all gates pass).
- Launcher emits correct flags; training initializes fully (Megatron 4 GPUs +
  SGLang 4 GPUs + session server + router + Harbor + terminus-2 all live).

### RESOLVED: use mini-swe-agent (not terminus-2) for Miles TITO

Initial attempts with **terminus-2** returned `AgentError`/`reward=0.0` from:

```
400 - rollback failed: no assistant message found in the first 1 matched messages
```

Root cause (`miles/rollout/session/linear_trajectory.py`): the TITO session
server is append-only and anchors checkpoints on **assistant** messages, so the
agent must resend the growing `[sys, user, assistant, tool, assistant, ...]`
history. terminus-2 instead sends `[system, user]` each turn (user = evolving
terminal observation) and never resends assistant messages -> prefix diverges
with no assistant checkpoint -> rejected. terminus-2 (CodeContests' SkyRL
packaging agent) is protocol-incompatible with Miles' on-policy TITO capture.

FIX (validated 2026-06-10): switch to **mini-swe-agent** + `codecontests.yaml`
(this dir). mini-swe-agent keeps append-only tool-call history -> TITO-compatible.
The CodeContests task is agent-agnostic (its `instruction.md` already states the
`/app/solution.py` contract), so `codecontests.yaml` just keeps mini-swe-agent's
tool-call/submit mechanics, sets `environment_class: local`, `cwd: /app`, and
folds in terminus-2's analysis->plan->act reasoning framing.

Wiring (both required):
- `agent_name=mini-swe-agent` in the prompt jsonl (`build_cc_jsonl.py` default).
- Start the Harbor server with
  `MSWEA_CONFIG_FILE=.../qwen3-codecontests/codecontests.yaml` and
  `OPENAI_API_KEY=dummy` (Harbor 0.13.1's mini_swe_agent adapter honors
  `config_file`; no adapter patch needed).

Results:
- Standalone single trial: `code_contests-0000` -> exit_status=Submitted, reward=1.0.
- Full async TITO training (20-task set): trials Submitted, reward=1.0, ZERO
  rollback errors; `update_weights_from_distributed` 200 OK (Megatron->SGLang
  weight sync via `postproc_lock`). Rollout -> reward -> GRPO -> weight broadcast
  all functional. Scale up by pointing `--prompt-data` at the full
  `~/cc_train.jsonl` (9644 tasks) and raising `--num-rollout`.

Live fixes already applied during bring-up (so they are not re-discovered):
- launcher `megatron_path` must be `/app/Megatron-LM` (not `/root/...`).
- Harbor server must be started with `OPENAI_API_KEY=dummy` in its env (terminus
  LiteLLM reads the key from env; api_base is wired per-request/session).
- The agent's LLM endpoint is the session-scoped URL
  `http://<router-host>:30000/sessions/{id}/v1/chat/completions` (port 30000 =
  session server, NOT 31000).
