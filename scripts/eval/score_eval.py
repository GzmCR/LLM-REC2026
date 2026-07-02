#!/usr/bin/env python3
"""Score local proxy predictions for LLM-Rec evaluation tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ITEMIC_RE = re.compile(r"<\|(?P<prefix>video|ad|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TASKS = ["item_understanding", "recommendation", "user_related_items", "user_logic_chain", "general_mcq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_itemics(text: str) -> list[str]:
    return [m.group(0) for m in ITEMIC_RE.finditer(text or "")]


def domain_of(token: str) -> str:
    m = ITEMIC_RE.search(token)
    return m.group("prefix") if m else "unknown"


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def parse_json_payload(text: str) -> Any | None:
    payload = strip_think(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def f1_sets(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    hit = len(pred & gold)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def prediction_map(pred_dir: Path, task: str) -> dict[str, dict[str, Any]]:
    return {rec["id"]: rec for rec in read_jsonl(pred_dir / f"{task}.jsonl")}


def score_item_understanding(eval_dir: Path, pred_dir: Path, out_dir: Path) -> dict[str, Any]:
    preds = prediction_map(pred_dir, "item_understanding")
    rows = []
    for rec in read_jsonl(eval_dir / "item_understanding.jsonl"):
        outputs = preds.get(rec["id"], {}).get("outputs", [])
        candidates = [tok for output in outputs for tok in extract_itemics(output)]
        unique = sorted(set(candidates))
        gold = set(rec.get("gold", []))
        passed = int(bool(gold & set(unique)))
        gold_prefix = domain_of(next(iter(gold))) if gold else "unknown"
        prefix_ok = int(any(domain_of(tok) == gold_prefix for tok in unique))
        rows.append({
            "id": rec["id"],
            "pass": passed,
            "num_outputs": len(outputs),
            "num_candidates": len(candidates),
            "num_unique_candidates": len(unique),
            "valid_candidate_rate": len(candidates) / len(outputs) if outputs else 0.0,
            "prefix_ok": prefix_ok,
        })
    write_csv(out_dir / "item_understanding.csv", rows)
    return {
        "count": len(rows),
        "pass@64": mean([r["pass"] for r in rows]),
        "valid_candidate_rate": mean([r["valid_candidate_rate"] for r in rows]),
        "unique_candidate_avg": mean([r["num_unique_candidates"] for r in rows]),
        "prefix_accuracy": mean([r["prefix_ok"] for r in rows]),
    }


def score_recommendation(eval_dir: Path, pred_dir: Path, out_dir: Path) -> dict[str, Any]:
    preds = prediction_map(pred_dir, "recommendation")
    rows = []
    token_rows = []
    for rec in read_jsonl(eval_dir / "recommendation.jsonl"):
        outputs = preds.get(rec["id"], {}).get("outputs", [])
        candidates = [tok for output in outputs for tok in extract_itemics(output)]
        cand_set = set(candidates)
        unique = set(candidates)
        gold = rec.get("gold", [])
        any_pass = int(bool(cand_set & set(gold)))
        for token in gold:
            token_rows.append({"domain": domain_of(token), "pass": int(token in cand_set)})
        rows.append({
            "id": rec["id"],
            "sample_any_pass": any_pass,
            "gold_count": len(gold),
            "num_outputs": len(outputs),
            "num_candidates": len(candidates),
            "num_unique_candidates": len(unique),
            "valid_candidate_rate": len(candidates) / len(outputs) if outputs else 0.0,
            "duplicate_rate": 1 - (len(unique) / len(candidates)) if candidates else 0.0,
        })
    write_csv(out_dir / "recommendation.csv", rows)
    by_domain = {}
    for domain in ["video", "prod", "living", "ad"]:
        vals = [r["pass"] for r in token_rows if r["domain"] == domain]
        by_domain[f"{domain}_pass@64"] = mean(vals)
        by_domain[f"{domain}_count"] = len(vals)
    return {
        "count": len(rows),
        "overall_micro_pass@64": mean([r["pass"] for r in token_rows]),
        "sample_any_pass@64": mean([r["sample_any_pass"] for r in rows]),
        "valid_candidate_rate": mean([r["valid_candidate_rate"] for r in rows]),
        "duplicate_rate": mean([r["duplicate_rate"] for r in rows]),
        **by_domain,
    }


def score_user_related(eval_dir: Path, pred_dir: Path, out_dir: Path) -> dict[str, Any]:
    preds = prediction_map(pred_dir, "user_related_items")
    rows = []
    for rec in read_jsonl(eval_dir / "user_related_items.jsonl"):
        outputs = preds.get(rec["id"], {}).get("outputs", [""])
        parsed = parse_json_payload(outputs[0] if outputs else "")
        json_ok = isinstance(parsed, list)
        pred_tokens = set(x for x in parsed if isinstance(x, str) and ITEMIC_RE.fullmatch(x)) if json_ok else set(extract_itemics(outputs[0] if outputs else ""))
        gold = set(rec.get("gold", []))
        history = set(rec.get("history_tokens", []))
        precision, recall, f1 = f1_sets(pred_tokens, gold)
        rows.append({
            "id": rec["id"],
            "json_ok": int(json_ok),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_count": len(pred_tokens),
            "gold_count": len(gold),
            "history_grounding_rate": len(pred_tokens & history) / len(pred_tokens) if pred_tokens else 0.0,
        })
    write_csv(out_dir / "user_interest_related.csv", rows)
    return {
        "count": len(rows),
        "precision": mean([r["precision"] for r in rows]),
        "recall": mean([r["recall"] for r in rows]),
        "f1": mean([r["f1"] for r in rows]),
        "json_parse_rate": mean([r["json_ok"] for r in rows]),
        "history_grounding_rate": mean([r["history_grounding_rate"] for r in rows]),
    }


def score_user_logic(eval_dir: Path, pred_dir: Path, out_dir: Path) -> dict[str, Any]:
    preds = prediction_map(pred_dir, "user_logic_chain")
    rows = []
    for rec in read_jsonl(eval_dir / "user_logic_chain.jsonl"):
        outputs = preds.get(rec["id"], {}).get("outputs", [""])
        parsed = parse_json_payload(outputs[0] if outputs else "")
        json_ok = isinstance(parsed, dict)
        events = []
        if json_ok and isinstance(parsed.get("logic_chain"), dict):
            events = parsed["logic_chain"].get("events") or []
        valid_events = isinstance(events, list) and 1 <= len(events) <= 5
        pred_text = json.dumps(events, ensure_ascii=False) if events else (outputs[0] if outputs else "")
        pred_tokens = set(extract_itemics(pred_text))
        gold_tokens = set(rec.get("gold_event_tokens", []))
        history = set(rec.get("history_tokens", []))
        _, _, token_f1 = f1_sets(pred_tokens, gold_tokens)
        date_ok = []
        logic_nonempty = []
        for event in events if isinstance(events, list) else []:
            date_ok.append(bool(DATE_RE.match(str(event.get("date", "")))) if isinstance(event, dict) else False)
            logic_nonempty.append(bool(str(event.get("logic", "")).strip()) if isinstance(event, dict) else False)
        grounding = len(pred_tokens & history) / len(pred_tokens) if pred_tokens else 0.0
        proxy_parts = [int(json_ok), int(valid_events), grounding, token_f1, mean([int(x) for x in date_ok]), mean([int(x) for x in logic_nonempty])]
        rows.append({
            "id": rec["id"],
            "json_ok": int(json_ok),
            "has_valid_events": int(valid_events),
            "event_count": len(events) if isinstance(events, list) else 0,
            "event_sid_f1": token_f1,
            "history_grounding_rate": grounding,
            "date_format_rate": mean([int(x) for x in date_ok]),
            "logic_nonempty_rate": mean([int(x) for x in logic_nonempty]),
            "proxy_score": mean(proxy_parts),
        })
    write_csv(out_dir / "user_interest_logic_chain.csv", rows)
    return {
        "count": len(rows),
        "json_parse_rate": mean([r["json_ok"] for r in rows]),
        "valid_events_rate": mean([r["has_valid_events"] for r in rows]),
        "event_sid_f1": mean([r["event_sid_f1"] for r in rows]),
        "history_grounding_rate": mean([r["history_grounding_rate"] for r in rows]),
        "date_format_rate": mean([r["date_format_rate"] for r in rows]),
        "logic_nonempty_rate": mean([r["logic_nonempty_rate"] for r in rows]),
        "proxy_score": mean([r["proxy_score"] for r in rows]),
    }


def normalize_mcq_answer(text: str) -> str:
    text = str(text).upper()
    m = re.search(r"正确答案\s*是\s*([A-Z][A-Z\s,，、]*)", text)
    if m:
        text = m.group(1)
    return "".join(sorted(set(re.findall(r"[A-Z]", text))))


def score_general_mcq(eval_dir: Path, pred_dir: Path, out_dir: Path) -> dict[str, Any]:
    preds = prediction_map(pred_dir, "general_mcq")
    rows = []
    for rec in read_jsonl(eval_dir / "general_mcq.jsonl"):
        outputs = preds.get(rec["id"], {}).get("outputs", [""])
        pred = normalize_mcq_answer(outputs[0] if outputs else "")
        gold = normalize_mcq_answer(rec.get("gold", ""))
        rows.append({"id": rec["id"], "pred": pred, "gold": gold, "correct": int(pred == gold)})
    write_csv(out_dir / "general_mcq.csv", rows)
    return {"count": len(rows), "accuracy": mean([r["correct"] for r in rows])}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Local Evaluation Summary", ""]
    for task, metrics in summary.items():
        lines.append(f"## {task}")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- `{key}`: {value:.6f}")
            else:
                lines.append(f"- `{key}`: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "item_understanding": score_item_understanding(eval_dir, pred_dir, out_dir),
        "recommendation": score_recommendation(eval_dir, pred_dir, out_dir),
        "user_related_items": score_user_related(eval_dir, pred_dir, out_dir),
        "user_logic_chain": score_user_logic(eval_dir, pred_dir, out_dir),
        "general_mcq": score_general_mcq(eval_dir, pred_dir, out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(out_dir / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
