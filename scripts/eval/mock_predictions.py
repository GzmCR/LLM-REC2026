#!/usr/bin/env python3
"""Create deterministic mock predictions for local scorer tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EVAL_FILES = [
    "item_understanding",
    "recommendation",
    "user_related_items",
    "user_logic_chain",
    "general_mcq",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def wrong_token() -> str:
    return "<|prod_begin|><s_a_0><s_b_0><s_c_0>"


def mock_output(task: str, rec: dict, idx: int) -> list[str]:
    hit = idx % 2 == 0
    if task in {"item_understanding", "recommendation"}:
        outputs = [wrong_token() for _ in range(64)]
        if hit and rec.get("gold"):
            outputs[3] = rec["gold"][0]
        return outputs
    if task == "user_related_items":
        return [json.dumps(rec.get("gold", []) if hit else [], ensure_ascii=False)]
    if task == "user_logic_chain":
        if hit:
            return [json.dumps(rec.get("gold", {}), ensure_ascii=False)]
        return ["not a json"]
    if task == "general_mcq":
        return [f"正确答案是 {rec.get('gold', '') if hit else 'Z'}"]
    return [""]


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in EVAL_FILES:
        out_path = out_dir / f"{task}.jsonl"
        with out_path.open("w", encoding="utf-8") as fp:
            for idx, rec in enumerate(read_jsonl(eval_dir / f"{task}.jsonl") or []):
                pred = {"id": rec["id"], "task": rec["task"], "outputs": mock_output(task, rec, idx)}
                fp.write(json.dumps(pred, ensure_ascii=False) + "\n")
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
