# Multi Turn RL using Qwen3-1.7B on CodeContests (Miles + Megatron + Harbor)

Self-contained example to RL-train **Qwen3-1.7B** (dense) on the **CodeContests**
Harbor dataset with Miles + Megatron + Harbor, using the **mini-swe-agent**.

The whole workflow is driven from the notebook
`[run_from_scratch.ipynb](run_from_scratch.ipynb)`: two containers on a shared
Docker network — `miles_swe` (Megatron trainer + SGLang rollout + TITO session
server) and `agent_env` (Harbor orchestrator + mini-swe-agent) — orchestrated from
a lightweight `cc_notebook` JupyterLab container.

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/sreerohi/miles_aai.git
cd miles_aai
```

### 2. Build the three images

Note the different build contexts:

```bash
docker build -f docker/Dockerfile.cc-base    docker/ -t miles_base:v1     # pulls the ~75GB rlsys base
docker build -f docker/Dockerfile.agent-base docker/ -t agent_base:v1
docker build -f docker/Dockerfile.notebook   .       -t cc_notebook:v1    # context = repo root
```

### 3. Launch the notebook container and drop into its shell

```bash
GPU=0 WANDB_KEY=<optional> bash docker/run_notebook.sh
```

This creates the `swe-net` network, mounts the host repo + runtime + Docker
socket, and execs you into a bash shell. Set `GPU` to an idle device (default is `7`).

### 4. Start JupyterLab from inside that shell

```bash
jupyter lab "examples/experimental/qwen3-codecontests/run_from_scratch.ipynb" --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Copy the printed `http://127.0.0.1:8888/lab?token=...` URL; open it in your browser
as `http://<host-ip>:8888/lab?token=...` (port 8888 is published).

### 5. Run the notebook cells top-to-bottom

- **Config cell** — derives all paths from the repo.
- **§1 Host setup** — creates `swe-net` (if not present).
- **§2** — launches `miles_swe` + `agent_env` from the images you built.
- **§3** — `pip install -e .` miles into `miles_swe`.
- **§4** — starts Harbor.
- **§5 Data prep** — downloads the CodeContests dataset (slow, one-time).
- **§6** — pre-fetches the Qwen3-1.7B weights.
- **§7+** — full training + the live monitors.

## Caveats worth knowing before you start

- The notebook **does not build** `miles_base` / `agent_base` — it only `docker run`s
them, so step 2 is mandatory first.
- Because the sibling containers bind-mount the host repo (`-v $REPO_DIR:$REPO_DIR`),
the `git clone` from step 1 is what makes them work.
- **GPU exposure:** the notebook starts `miles_swe` with `--privileged --ipc host`,
which gives ROCm device access; ensure the host ROCm stack is healthy (`rocm-smi` works).
- **HF:** Qwen3-1.7B and `open-thoughts/CodeContests` are public, so no token is
normally needed; set `HF_TOKEN` only if you hit rate limits.

