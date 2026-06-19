#!/usr/bin/env python3
"""Build the Miles prompt JSONL from extracted CodeContests Harbor task dirs.

Walks ``<tasks>/`` (output of extract_codecontests.py), reading each task's
``instruction.md`` as the prompt and emitting one Miles sample per task:

    {"prompt": <instruction.md text>,
     "metadata": {"instance_id": <dir name>, "agent_name": "terminus-2", "split": "train"}}

CRITICAL: ``metadata.instance_id`` MUST equal the task directory name, because
``server.py`` resolves the task under ``HARBOR_TASKS_DIR`` by that id. A mismatch
yields ``TaskNotFound`` and silent 0.0 rewards.

Usage:
    python build_cc_jsonl.py --tasks ~/harbor_tasks_cc --out ~/cc_train.jsonl
    python build_cc_jsonl.py --tasks ~/harbor_tasks_cc --out ~/cc_train_small.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build(
    tasks_dir: str,
    out_path: str,
    *,
    agent_name: str = "terminus-2",
    split: str = "train",
    limit: int | None = None,
) -> int:
    tasks = Path(os.path.expanduser(tasks_dir))
    if not tasks.is_dir():
        raise FileNotFoundError(f"tasks dir not found: {tasks}")

    out = Path(os.path.expanduser(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w") as fout:
        for d in sorted(p for p in tasks.iterdir() if p.is_dir()):
            instruction = d / "instruction.md"
            if not instruction.is_file():
                continue
            prompt = instruction.read_text(errors="replace").strip()
            if not prompt:
                continue
            sample = {
                "prompt": prompt,
                "metadata": {
                    "instance_id": d.name,  # MUST match the task dir name
                    "agent_name": agent_name,
                    "split": split,
                },
            }
            fout.write(json.dumps(sample) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    print(f"wrote {count} samples -> {out}")
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="CodeContests task dirs -> Miles prompt JSONL")
    ap.add_argument("--tasks", default=os.path.expanduser("~/harbor_tasks_cc"))
    ap.add_argument("--out", default=os.path.expanduser("~/cc_train.jsonl"))
    ap.add_argument("--agent-name", default="terminus-2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(
        args.tasks,
        args.out,
        agent_name=args.agent_name,
        split=args.split,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
