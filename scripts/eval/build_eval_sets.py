#!/usr/bin/env python3
"""Build local proxy evaluation sets for the LLM-Rec competition tasks."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ITEMIC_RE = re.compile(r"<\|(?P<prefix>video|ad|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
TASK_FILES = {
    "item_understanding": ["懂物料part1.jsonl", "懂物料part2.jsonl", "懂物料part3.jsonl", "懂物料part4.jsonl"],
    "recommendation": ["懂推荐1.jsonl", "懂推荐2.jsonl", "懂推荐3.jsonl", "懂推荐4.jsonl"],
    "user_interest": ["懂用户.jsonl"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed local eval sets.")
    parser.add_argument("--sft-dir", default="dataset")
    parser.add_argument("--mcq-path", default=None)
    parser.add_argument("--out-dir", default="outputs/eval/eval_sets")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-ratio", type=float, default=0.08)
    parser.add_argument("--max-per-task", type=int, default=1000)
    return parser.parse_args()


def load_sft_record(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list):
        if not obj:
            return None
        obj = obj[0]
    return obj if isinstance(obj, dict) else None


def messages_from_sft(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    system = record.get("system") or ""
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": record.get("prompt") or ""})
    return messages


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def extract_itemics(text: str) -> list[str]:
    return [m.group(0) for m in ITEMIC_RE.finditer(text or "")]


def domain_of(token: str) -> str:
    m = ITEMIC_RE.search(token)
    return m.group("prefix") if m else "unknown"


def sample_records(records: list[dict[str, Any]], rng: random.Random, eval_ratio: float, max_per_task: int) -> list[dict[str, Any]]:
    if not records:
        return []
    count = max(1, int(round(len(records) * eval_ratio)))
    if max_per_task > 0:
        count = min(count, max_per_task)
    count = min(count, len(records))
    shuffled = list(records)
    rng.shuffle(shuffled)
    return sorted(shuffled[:count], key=lambda r: (r["source_file"], r["line_no"]))


def iter_sft_files(sft_dir: Path, names: list[str]):
    for name in names:
        path = sft_dir / name
        if path.exists():
            yield path


def build_item_understanding(sft_dir: Path, rng: random.Random, eval_ratio: float, max_per_task: int) -> list[dict[str, Any]]:
    records = []
    for path in iter_sft_files(sft_dir, TASK_FILES["item_understanding"]):
        with path.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                obj = load_sft_record(line)
                if not obj:
                    continue
                gold = extract_itemics(obj.get("response") or "")
                if not gold:
                    continue
                records.append({
                    "id": f"{path.name}:{line_no}",
                    "task": "item_understanding",
                    "messages": messages_from_sft(obj),
                    "gold": [gold[0]],
                    "source_file": path.name,
                    "line_no": line_no,
                })
    return sample_records(records, rng, eval_ratio, max_per_task)


def build_recommendation(sft_dir: Path, rng: random.Random, eval_ratio: float, max_per_task: int) -> list[dict[str, Any]]:
    records = []
    for path in iter_sft_files(sft_dir, TASK_FILES["recommendation"]):
        with path.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                obj = load_sft_record(line)
                if not obj:
                    continue
                gold = sorted(set(extract_itemics(obj.get("response") or "")))
                if not gold:
                    continue
                by_domain: dict[str, list[str]] = defaultdict(list)
                for token in gold:
                    by_domain[domain_of(token)].append(token)
                records.append({
                    "id": f"{path.name}:{line_no}",
                    "task": "recommendation",
                    "messages": messages_from_sft(obj),
                    "gold": gold,
                    "gold_by_domain": dict(by_domain),
                    "source_file": path.name,
                    "line_no": line_no,
                })
    return sample_records(records, rng, eval_ratio, max_per_task)


def parse_response_payload(response: str) -> Any | None:
    payload = strip_think(response)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def build_user_interest(sft_dir: Path, rng: random.Random, eval_ratio: float, max_per_task: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    related = []
    logic = []
    for path in iter_sft_files(sft_dir, TASK_FILES["user_interest"]):
        with path.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                obj = load_sft_record(line)
                if not obj:
                    continue
                prompt = obj.get("prompt") or ""
                payload = parse_response_payload(obj.get("response") or "")
                base = {
                    "id": f"{path.name}:{line_no}",
                    "messages": messages_from_sft(obj),
                    "history_tokens": sorted(set(extract_itemics(prompt))),
                    "source_file": path.name,
                    "line_no": line_no,
                }
                if isinstance(payload, list):
                    gold = sorted(set(x for x in payload if isinstance(x, str) and ITEMIC_RE.fullmatch(x)))
                    if gold:
                        related.append({**base, "task": "user_related_items", "gold": gold})
                elif isinstance(payload, dict) and isinstance(payload.get("logic_chain"), dict):
                    events = payload["logic_chain"].get("events") or []
                    gold_tokens = sorted(set(extract_itemics(json.dumps(events, ensure_ascii=False))))
                    logic.append({
                        **base,
                        "task": "user_logic_chain",
                        "gold": payload,
                        "gold_event_tokens": gold_tokens,
                    })
    return (
        sample_records(related, rng, eval_ratio, max_per_task),
        sample_records(logic, rng, eval_ratio, max_per_task),
    )


def normalize_answer(answer: str) -> str:
    return "".join(sorted(set(re.findall(r"[A-Z]", str(answer).upper()))))


def iter_mcq_records(path: Path):
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "messages" in obj and "metadata" in obj:
                yield line_no, obj
                continue
            if isinstance(obj, dict):
                for value in obj.values():
                    if isinstance(value, dict) and "messages" in value and "metadata" in value:
                        yield line_no, value


def build_general_mcq(mcq_path: str | None, rng: random.Random, eval_ratio: float, max_per_task: int) -> list[dict[str, Any]]:
    if not mcq_path:
        return []
    path = Path(mcq_path)
    if not path.exists():
        return []
    records = []
    for line_no, obj in iter_mcq_records(path):
        answer = normalize_answer(obj.get("metadata", {}).get("answer", ""))
        if not answer:
            continue
        records.append({
            "id": f"{path.name}:{line_no}:{len(records)+1}",
            "task": "general_mcq",
            "messages": obj["messages"],
            "gold": answer,
            "source_file": path.name,
            "line_no": line_no,
        })
    return sample_records(records, rng, eval_ratio, max_per_task)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    sft_dir = Path(args.sft_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_sets = {
        "item_understanding": build_item_understanding(sft_dir, rng, args.eval_ratio, args.max_per_task),
        "recommendation": build_recommendation(sft_dir, rng, args.eval_ratio, args.max_per_task),
    }
    related, logic = build_user_interest(sft_dir, rng, args.eval_ratio, args.max_per_task)
    eval_sets["user_related_items"] = related
    eval_sets["user_logic_chain"] = logic
    eval_sets["general_mcq"] = build_general_mcq(args.mcq_path, rng, args.eval_ratio, args.max_per_task)

    exclusions: dict[str, list[int]] = defaultdict(list)
    summary = {"seed": args.seed, "eval_ratio": args.eval_ratio, "max_per_task": args.max_per_task, "tasks": {}}
    for name, records in eval_sets.items():
        write_jsonl(out_dir / f"{name}.jsonl", records)
        summary["tasks"][name] = len(records)
        for rec in records:
            if rec.get("source_file") and rec.get("line_no"):
                exclusions[rec["source_file"]].append(int(rec["line_no"]))

    (out_dir / "train_exclusion_ids.json").write_text(
        json.dumps({k: sorted(set(v)) for k, v in exclusions.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
