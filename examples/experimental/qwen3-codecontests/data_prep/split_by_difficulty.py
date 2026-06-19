#!/usr/bin/env python3
"""Split CodeContests Harbor tasks into per-difficulty Miles JSONL files.

Difficulty comes from the Codeforces ``Rating`` field in each task's
``instruction.md`` (the only real per-problem difficulty signal; the
``difficulty`` field in task.toml is a constant "medium" and is ignored).
Tasks whose Rating is missing or 0 go to the ``unrated`` bucket.

Buckets (by Codeforces rating):
    easy        : rating <  1200
    easy_medium : 1200 <= rating < 1600
    medium      : 1600 <= rating < 2000
    hard        : 2000 <= rating < 2400
    very_hard   : rating >= 2400
    unrated     : no Rating / Rating == 0

Each output line is a Miles sample:
    {"prompt": <instruction.md>,
     "metadata": {"instance_id": <dir>, "agent_name": "mini-swe-agent",
                  "split": "train", "rating": <int|null>, "difficulty": <bucket>}}

Usage:
    python split_by_difficulty.py --tasks ~/harbor_tasks_cc --out-dir ./data
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

BUCKETS = ["easy", "easy_medium", "medium", "hard", "very_hard", "unrated"]
_RATING_RE = re.compile(r"\*\*Rating\*\*:\s*(\d+)")


def bucket_for(rating: int | None) -> str:
    if not rating:  # None or 0
        return "unrated"
    if rating < 1200:
        return "easy"
    if rating < 1600:
        return "easy_medium"
    if rating < 2000:
        return "medium"
    if rating < 2400:
        return "hard"
    return "very_hard"


def parse_rating(instruction_text: str) -> int | None:
    m = _RATING_RE.search(instruction_text)
    if not m:
        return None
    r = int(m.group(1))
    return r if r > 0 else None


def split(tasks_dir: str, out_dir: str, *, agent_name: str = "mini-swe-agent", split: str = "train") -> dict[str, int]:
    tasks = Path(os.path.expanduser(tasks_dir))
    if not tasks.is_dir():
        raise FileNotFoundError(f"tasks dir not found: {tasks}")
    out = Path(os.path.expanduser(out_dir))
    out.mkdir(parents=True, exist_ok=True)

    files = {b: open(out / f"cc_train_{b}.jsonl", "w") for b in BUCKETS}
    counts = {b: 0 for b in BUCKETS}
    try:
        for d in sorted(p for p in tasks.iterdir() if p.is_dir()):
            instr = d / "instruction.md"
            if not instr.is_file():
                continue
            text = instr.read_text(errors="replace")
            prompt = text.strip()
            if not prompt:
                continue
            rating = parse_rating(text)
            b = bucket_for(rating)
            sample = {
                "prompt": prompt,
                "metadata": {
                    "instance_id": d.name,
                    "agent_name": agent_name,
                    "split": split,
                    "rating": rating,
                    "difficulty": b,
                },
            }
            files[b].write(json.dumps(sample) + "\n")
            counts[b] += 1
    finally:
        for f in files.values():
            f.close()

    total = sum(counts.values())
    print(f"wrote {total} samples across {len(BUCKETS)} buckets -> {out}")
    for b in BUCKETS:
        print(f"  cc_train_{b}.jsonl : {counts[b]}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Split CodeContests tasks into per-difficulty JSONLs")
    ap.add_argument("--tasks", default=os.path.expanduser("~/harbor_tasks_cc"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--agent-name", default="mini-swe-agent")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    split(args.tasks, args.out_dir, agent_name=args.agent_name, split=args.split)


if __name__ == "__main__":
    main()
