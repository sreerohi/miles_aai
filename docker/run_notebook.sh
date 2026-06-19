#!/usr/bin/env bash
# Launch the qwen3-codecontests notebook-driver container and drop into a bash
# shell inside it . From that shell you start
# JupyterLab yourself, e.g.:
#     jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
# then open http://<host>:8888 with the token it prints.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (base)
EX="$REPO_DIR/examples/experimental/qwen3-codecontests"                     # dir you bash into
RUNTIME="$EX/runtime"
WORK_DIR="${WORK_DIR:-$RUNTIME/work}"                                       # logs + Harbor trials
TASKS_DIR="${TASKS_DIR:-$RUNTIME/harbor_tasks_cc}"                          # extracted Harbor task dirs
HF_CACHE="${HF_CACHE:-$RUNTIME/cache/hf}"                                   # HuggingFace cache (overridable)
GPU="${GPU:-7}"                                                             # CC_HIP_VISIBLE_DEVICES
WANDB_KEY="${WANDB_KEY:-}"                                                  # optional; empty => W&B disabled
IMAGE="${IMAGE:-cc_notebook:v1}"
NAME="${NAME:-cc_notebook}"
PORT="${PORT:-8888}"

mkdir -p "$WORK_DIR" "$TASKS_DIR" "$HF_CACHE"
docker network inspect swe-net >/dev/null 2>&1 || docker network create swe-net

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --hostname "$NAME" --network swe-net \
    -p "${PORT}:8888" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$REPO_DIR":"$REPO_DIR" \
    -v "$HF_CACHE":"$HF_CACHE" \
    -e REPO_DIR="$REPO_DIR" \
    -e EX="$EX" \
    -e WORK_DIR="$WORK_DIR" \
    -e TASKS_DIR="$TASKS_DIR" \
    -e GPU="$GPU" \
    -e WANDB_KEY="$WANDB_KEY" \
    "$IMAGE"
fi

NB="examples/experimental/qwen3-codecontests/run_from_scratch.ipynb"
echo "Container '$NAME' is up (JupyterLab port ${PORT} published). You are now in: $REPO_DIR"
echo "Open the notebook directly with:"
echo "    jupyter lab \"$NB\" --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
echo "then in your browser go to:  http://<this-host>:${PORT}/lab/tree/$NB?token=<token-from-the-output>"
exec docker exec -it -w "$REPO_DIR" "$NAME" bash
