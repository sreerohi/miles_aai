#!/usr/bin/env bash
# Launch the SINGLE-IMAGE qwen3-codecontests workshop container and drop into a
# bash shell. One container runs everything: Megatron/SGLang trainer, the Harbor
# agent server, and JupyterLab. Harbor grades tasks on the bare host via the
# `subprocess` (bubblewrap) environment -- NO docker socket, NO swe-net, NO
# per-task containers.
#
# From the shell, start JupyterLab yourself:
#     jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
# then open http://<host>:8888 with the token it prints.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EX="$REPO_DIR/examples/experimental/qwen3-codecontests"
RUNTIME="$EX/runtime"
WORK_DIR="${WORK_DIR:-$RUNTIME/work}"          # logs + Harbor trials (cc_trials/)
HF_CACHE="${HF_CACHE:-$RUNTIME/cache/hf}"      # HuggingFace cache (model weights)
GPU="${GPU:-7}"                                # CC_HIP_VISIBLE_DEVICES
WANDB_KEY="${WANDB_KEY:-}"                      # optional; empty => W&B disabled
IMAGE="${IMAGE:-miles_workshop:v1}"
NAME="${NAME:-cc_workshop}"
PORT="${PORT:-8888}"
HARBOR_PORT="${HARBOR_PORT:-11000}"

mkdir -p "$WORK_DIR" "$HF_CACHE"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # GPU + SGLang needs: device access, --ipc host, large shm. The repo is
  # identity-mounted (host path == container path) so trial/work paths written
  # into trials resolve the same inside and out. No docker.sock, no swe-net.
  docker run -d --name "$NAME" --hostname "$NAME" \
    --privileged --ipc host --shm-size 32g --security-opt label=disable \
    -p "${PORT}:8888" \
    -p "${HARBOR_PORT}:11000" \
    -v "$REPO_DIR":"$REPO_DIR" \
    -v "$HF_CACHE":"$HF_CACHE" \
    -e REPO_DIR="$REPO_DIR" \
    -e EX="$EX" \
    -e WORK_DIR="$WORK_DIR" \
    -e HF_HOME="$HF_CACHE" \
    -e GPU="$GPU" \
    -e WANDB_KEY="$WANDB_KEY" \
    -e HARBOR_TRIALS_DIR="$WORK_DIR/cc_trials" \
    -e MILES_HOST_IP="${MILES_HOST_IP:-127.0.0.1}" \
    -e MILES_ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST:-127.0.0.1}" \
    -e HARBOR_SANDBOX_BACKEND="${HARBOR_SANDBOX_BACKEND:-proot}" \
    "$IMAGE"
fi

NB="examples/experimental/qwen3-codecontests/run_from_scratch_single.ipynb"
echo "Container '$NAME' is up (Jupyter :${PORT}, Harbor :${HARBOR_PORT})."
echo "Open the notebook with:"
echo "    jupyter lab \"$NB\" --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
echo "then browse to:  http://<this-host>:${PORT}/lab/tree/$NB?token=<token>"
exec docker exec -it -w "$REPO_DIR" "$NAME" bash
