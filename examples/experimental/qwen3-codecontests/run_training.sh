#!/usr/bin/env bash
# Multi-turn RL on MILES — Qwen3-1.7B on CodeContests, from a plain shell.
#
# Shell-runnable equivalent of run_from_scratch.ipynb. Brings up the two
# containers (miles_swe trainer + agent_env Harbor), installs miles, starts
# Harbor, prepares data, pre-fetches the model, then runs the GRPO training
# loop in the foreground (streaming to the terminal and to $WORK_DIR/train.log).
#
# Prereqs (NOT done here — same as the notebook): images miles_base:v1 and
# agent_base:v1 must already be built, and an idle GPU is available.
#
# Usage:
#   bash run_training.sh                 # full flow end-to-end
#   GPU=0 WANDB_KEY=xxx bash run_training.sh
#   bash run_training.sh --reset         # just reset the trainer between runs, then exit
#   bash run_training.sh --skip-setup    # reuse running containers / installed miles
#   bash run_training.sh --skip-data     # reuse already-prepared data + model
#   bash run_training.sh --help
#
# Any flags after `--` are passed straight through to run-qwen3-codecontests.py,
# e.g.:  bash run_training.sh -- --num-rollout 50 --save-interval 5
set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. Config — all paths derived from the repo; overridable via environment.
# --------------------------------------------------------------------------- #
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
EX="${EX:-$REPO_DIR/examples/experimental/qwen3-codecontests}"
RUNTIME="${RUNTIME:-$EX/runtime}"
WORK_DIR="${WORK_DIR:-$RUNTIME/work}"            # logs + Harbor trials (cc_trials/)
TASKS_DIR="${TASKS_DIR:-$RUNTIME/harbor_tasks_cc}"  # extracted Harbor task dirs
HF_CACHE="${HF_CACHE:-$RUNTIME/cache/hf}"        # HuggingFace cache (can be large)
GPU="${GPU:-7}"                                   # CC_HIP_VISIBLE_DEVICES (pin one idle GPU)
WANDB_KEY="${WANDB_KEY:-}"                         # optional; empty => W&B disabled
PROMPT_DATA="${PROMPT_DATA:-$EX/data/cc_train_easy.jsonl}"  # curriculum: easy first

# Training hyperparameters (overridable via env or passthrough flags after --).
NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

SKIP_SETUP=0
SKIP_DATA=0
RESET_ONLY=0
PASSTHRU=()

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-data)  SKIP_DATA=1; shift ;;
    --reset)      RESET_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --) shift; PASSTHRU=("$@"); break ;;
    *) echo "unknown arg: $1 (use --help)"; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
note() { printf '\033[2m%s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------- #
# reset: restart the trainer to clear ray/sglang clusters + zombie workers
# (pkill alone leaves them, which stacks runs -> endpoint 500/404 errors).
# Keeps the image and the `pip install -e` (writable layer survives restart).
# --------------------------------------------------------------------------- #
reset_trainer() {
  log "Reset trainer (restart miles_swe)"
  docker restart miles_swe
  docker exec miles_swe bash -lc 'rm -rf /tmp/ray/session_* 2>/dev/null; rm -f /app/aiter/aiter/jit/build/lock_* 2>/dev/null' || true
  echo "miles_swe reset; live ray/sglang: $(docker exec miles_swe bash -lc 'pgrep -fc "raylet|sglang|train.py" 2>/dev/null' || echo 0)"
}

if [ "$RESET_ONLY" = 1 ]; then
  reset_trainer
  exit 0
fi

export REPO_DIR EX RUNTIME WORK_DIR TASKS_DIR HF_CACHE GPU WANDB_KEY
mkdir -p "$WORK_DIR" "$TASKS_DIR" "$HF_CACHE"

log "Config"
printf '%-12s= %s\n' REPO_DIR "$REPO_DIR" EX "$EX" WORK_DIR "$WORK_DIR" \
  TASKS_DIR "$TASKS_DIR" HF_CACHE "$HF_CACHE" GPU "$GPU" PROMPT_DATA "$PROMPT_DATA"
echo "WANDB_KEY   = $([ -n "$WANDB_KEY" ] && echo set || echo '(empty -> W&B disabled)')"

# --------------------------------------------------------------------------- #
# 1+2+3. Host setup, launch containers, install miles
# --------------------------------------------------------------------------- #
if [ "$SKIP_SETUP" = 0 ]; then
  log "1. Host setup & clean slate"
  docker rm -f miles_swe agent_env 2>/dev/null || true
  docker rm -f $(docker ps -aq --filter "name=code_contests-") 2>/dev/null || true
  docker network create swe-net 2>/dev/null || echo "swe-net exists"

  log "2. Launch containers (miles_swe + agent_env)"
  # Repo is IDENTITY-mounted (host path == container path) so work/tasks/trials
  # resolve in the sibling task containers Harbor spawns. HF cache mounted
  # separately and exposed via HF_HOME.
  docker run -d --name miles_swe --hostname miles_swe \
    --network swe-net \
    --privileged --ipc host --shm-size 32g --security-opt label=disable \
    -v "$REPO_DIR":"$REPO_DIR" \
    -v "$HF_CACHE":"$HF_CACHE" \
    -e HF_HOME="$HF_CACHE" \
    miles_base:v1 sleep infinity

  docker run -d --name agent_env --hostname agent_env \
    --network swe-net \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$REPO_DIR":"$REPO_DIR" \
    agent_base:v1 sleep infinity

  docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' | grep -E 'miles_swe|agent_env'

  log "3. Install miles in miles_swe"
  docker exec -e REPO_DIR="$REPO_DIR" miles_swe bash -lc 'cd "$REPO_DIR" && pip install -e . --no-deps --no-build-isolation'
  docker exec miles_swe bash -lc 'python3 -c "import miles, miles_plugins.mbridge"' && echo "miles import ok"
else
  note "--skip-setup: reusing running containers + installed miles"
fi

# --------------------------------------------------------------------------- #
# 4. Start Harbor (detached background service); wait for /health
# --------------------------------------------------------------------------- #
log "4. Start Harbor (background)"
if docker exec miles_swe bash -lc 'curl -sf http://agent_env:11000/health' >/dev/null 2>&1; then
  note "Harbor already responding; not relaunching"
else
  docker exec -d -e EX="$EX" -e REPO_DIR="$REPO_DIR" -e WORK_DIR="$WORK_DIR" -e TASKS_DIR="$TASKS_DIR" agent_env bash -lc ' \
      export OPENAI_API_KEY=dummy \
      MSWEA_API_KEY=dummy \
      MSWEA_CONFIG_FILE=$EX/harbor/codecontests.yaml \
      HARBOR_EXTRA_DOCKER_COMPOSE=$EX/harbor/swe_net_override.yaml \
      HARBOR_DELETE_CONTAINERS=true \
      HARBOR_TASKS_DIR=$TASKS_DIR \
      HARBOR_TRIALS_DIR=$WORK_DIR/cc_trials; \
      cd $EX && PYTHONPATH=$REPO_DIR python3 harbor/server.py --port 11000 --max-concurrent 8 > $WORK_DIR/harbor.log 2>&1'

  # health check runs from inside miles_swe (host can't reach swe-net)
  if docker exec miles_swe bash -lc 'for i in $(seq 1 30); do curl -sf http://agent_env:11000/health && exit 0; sleep 2; done; exit 1'; then
    echo " <- harbor up"
  else
    echo "harbor NOT up; last 15 lines of harbor.log:"; tail -n 15 "$WORK_DIR/harbor.log" 2>/dev/null || true
    exit 1
  fi
fi

# --------------------------------------------------------------------------- #
# 5+6. Data prep + model pre-fetch (guarded; no-op if already present)
# --------------------------------------------------------------------------- #
if [ "$SKIP_DATA" = 0 ]; then
  log "5. Data preparation (download + extract + split; guarded)"
  docker exec -e EX="$EX" -e TASKS_DIR="$TASKS_DIR" miles_swe bash -lc '
    [ -d "$TASKS_DIR"/code_contests-0000 ] || \
      python3 "$EX"/data_prep/extract_codecontests.py --dataset open-thoughts/CodeContests --out "$TASKS_DIR"
    [ -f "$EX"/data/cc_train_easy.jsonl ] || \
      python3 "$EX"/data_prep/split_by_difficulty.py --tasks "$TASKS_DIR" --out-dir "$EX"/data
  '

  log "6. Pre-fetch Qwen3-1.7B into HF cache"
  docker exec miles_swe bash -lc 'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-1.7B\")"'
else
  note "--skip-data: reusing prepared data + model"
fi

# --------------------------------------------------------------------------- #
# 7. Training run (GRPO loop) — foreground, streamed to terminal + train.log.
# Run ONE rollout/training at a time; use `--reset` between runs.
# --------------------------------------------------------------------------- #
log "7. Training run (foreground; also logging to $WORK_DIR/train.log)"
docker exec -e EX="$EX" -e REPO_DIR="$REPO_DIR" -e WORK_DIR="$WORK_DIR" -e TASKS_DIR="$TASKS_DIR" \
  -e GPU="$GPU" -e WANDB_KEY="$WANDB_KEY" \
  -e NUM_ROLLOUT="$NUM_ROLLOUT" -e ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
  -e N_SAMPLES_PER_PROMPT="$N_SAMPLES_PER_PROMPT" -e GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  -e MAX_SEQ_LEN="$MAX_SEQ_LEN" -e SAVE_INTERVAL="$SAVE_INTERVAL" -e PROMPT_DATA="$PROMPT_DATA" \
  miles_swe bash -lc 'cd "$EX"; \
    export CC_HIP_VISIBLE_DEVICES=$GPU \
    WANDB_KEY=$WANDB_KEY \
    MILES_ROUTER_EXTERNAL_HOST=miles_swe \
    AGENT_SERVER_URL=http://agent_env:11000 \
    HARBOR_TASKS_DIR=$TASKS_DIR \
    WANDB_DIR=$WORK_DIR/wandb; \
    PYTHONPATH=$REPO_DIR python3 run-qwen3-codecontests.py \
    --prompt-data "$PROMPT_DATA" \
    --num-rollout "$NUM_ROLLOUT" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" --save-interval "$SAVE_INTERVAL" '"${PASSTHRU[*]:-}"' 2>&1' \
  | tee "$WORK_DIR/train.log"

log "Training finished. Trials under $WORK_DIR/cc_trials  |  log: $WORK_DIR/train.log"
