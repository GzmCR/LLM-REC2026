#!/usr/bin/env python3
"""Register prepared train_data JSONL files in LLaMA-Factory dataset_info.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def alpaca_entry(path: Path) -> dict:
    return {
        "file_name": str(path.resolve()),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-factory-dir", default="third_party/LLaMA-Factory")
    parser.add_argument("--train-data-dir", default="train_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info_path = Path(args.llama_factory_dir) / "data" / "dataset_info.json"
    train_dir = Path(args.train_data_dir)

    if not info_path.exists():
        raise FileNotFoundError(f"dataset_info.json not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    entries = {
        "onereason_sft_official": train_dir / "data_official.jsonl",
        "onereason_sft_augmented": train_dir / "data_augmented.jsonl",
        "onereason_sft_mixed": train_dir / "data_mixed.jsonl",
    }

    for name, path in entries.items():
        if path.exists() and path.stat().st_size > 0:
            info[name] = alpaca_entry(path)
            print(f"[OK] registered {name}: {path.resolve()}")
        else:
            print(f"[skip] {name}: missing or empty ({path})")

    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] updated {info_path.resolve()}")


if __name__ == "__main__":
    main()
