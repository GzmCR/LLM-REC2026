#!/usr/bin/env python3
"""Build general MCQ SFT and eval JSONL from local CMMLU/MMLU files."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


LETTERS = "ABCD"
SYSTEM_TEMPLATES = [
    "你是一个非常聪明的助手，请直接遵循指示作答。",
    "你是一个严谨的选择题答题助手，请只输出题目要求的答案格式。",
    "你擅长常识、学科知识和逻辑推理，请根据题目选择正确选项。",
    "你需要完成单项选择题，请认真比较选项并给出最终答案。",
    "你是通用知识问答助手，请按指定格式返回正确选项。",
    "You are a careful multiple-choice QA assistant. Follow the requested answer format exactly.",
]

PROMPT_TEMPLATES = [
    "请回答以下问题（单项选择题）：\n\n{body}\n\n请按以下格式作答：\"正确答案是 (在此处填写选项字母)\"",
    "下面是一道单选题，请从 A、B、C、D 中选择唯一正确答案。\n\n{body}\n\n输出格式必须为：\"正确答案是 X\"",
    "请判断这道题的正确选项。\n\n{body}\n\n只需要回答：\"正确答案是 X\"",
    "请完成选择题。注意只能选择一个选项。\n\n{body}\n\n请按格式输出：正确答案是 X",
    "Answer the following multiple-choice question. Choose exactly one option.\n\n{body}\n\nReply in this exact format: \"正确答案是 X\"",
    "根据题干和选项给出最终答案。\n\n{body}\n\n不要解释，直接输出：\"正确答案是 X\"",
]


@dataclass(frozen=True)
class MCQSample:
    sample_id: str
    source: str
    split: str
    language: str
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build general MCQ SFT and eval JSONL from CMMLU/MMLU.")
    parser.add_argument("--cmmlu-dir", default="dataset/CMMLU")
    parser.add_argument("--mmlu-dir", default="dataset/MMLU")
    parser.add_argument("--train-out", default="generated_dataset/general_mcq_aux.jsonl")
    parser.add_argument("--eval-out", default="data_eval/general_mcq.jsonl")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means all train samples.")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 means all eval samples.")
    parser.add_argument("--language-mode", choices=["mixed", "zh", "en"], default="mixed")
    return parser.parse_args()


def answer_to_letter(answer: Any) -> str:
    if isinstance(answer, str):
        raw = answer.strip().upper()
        if raw in LETTERS:
            return raw
        if raw.isdigit():
            idx = int(raw)
            return LETTERS[idx] if 0 <= idx < 4 else ""
        return ""
    try:
        idx = int(answer)
    except (TypeError, ValueError):
        return ""
    return LETTERS[idx] if 0 <= idx < 4 else ""


def clean_text(value: Any) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def make_body(sample: MCQSample) -> str:
    lines = [sample.question]
    for letter, choice in zip(LETTERS, sample.choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def is_valid_sample(sample: MCQSample) -> bool:
    return (
        bool(sample.question)
        and sample.answer in LETTERS
        and len(sample.choices) == 4
        and all(bool(choice) for choice in sample.choices)
    )


def iter_mmlu(mmlu_dir: Path, split: str, skip: Counter[str]) -> Iterable[MCQSample]:
    split_to_pattern = {
        "train": "all/auxiliary_train-*.parquet",
        "eval": "all/validation-*.parquet",
    }
    for path in sorted(mmlu_dir.glob(split_to_pattern[split])):
        df = pd.read_parquet(path)
        for row_idx, row in df.iterrows():
            choices = row.get("choices")
            if choices is None:
                skip["mmlu_missing_choices"] += 1
                continue
            choice_list = [clean_text(x) for x in list(choices)]
            answer = answer_to_letter(row.get("answer"))
            sample = MCQSample(
                sample_id=f"mmlu:{split}:{row.get('subject') or 'unknown'}:{path.name}:{row_idx}",
                source="mmlu",
                split=split,
                language="en",
                subject=clean_text(row.get("subject") or ""),
                question=clean_text(row.get("question") or ""),
                choices=tuple(choice_list[:4]),  # type: ignore[arg-type]
                answer=answer,
            )
            if is_valid_sample(sample):
                yield sample
            else:
                skip["mmlu_invalid_sample"] += 1


def find_cmmlu_zip(cmmlu_dir: Path) -> Path | None:
    preferred = cmmlu_dir / "cmmlu_v1_0_1.zip"
    if preferred.exists():
        return preferred
    matches = sorted(cmmlu_dir.glob("*.zip"))
    return matches[0] if matches else None


def iter_cmmlu(cmmlu_dir: Path, split: str, skip: Counter[str]) -> Iterable[MCQSample]:
    zip_path = find_cmmlu_zip(cmmlu_dir)
    if not zip_path:
        skip["cmmlu_zip_missing"] += 1
        return
    zip_split = "test" if split == "train" else "dev"
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(name for name in zf.namelist() if name.startswith(f"{zip_split}/") and name.endswith(".csv"))
        for name in names:
            subject = Path(name).stem
            with zf.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text)
                for row_idx, row in enumerate(reader):
                    sample = MCQSample(
                        sample_id=f"cmmlu:{split}:{subject}:{row_idx}",
                        source="cmmlu",
                        split=split,
                        language="zh",
                        subject=subject,
                        question=clean_text(row.get("Question") or ""),
                        choices=(
                            clean_text(row.get("A") or ""),
                            clean_text(row.get("B") or ""),
                            clean_text(row.get("C") or ""),
                            clean_text(row.get("D") or ""),
                        ),
                        answer=answer_to_letter(row.get("Answer")),
                    )
                    if is_valid_sample(sample):
                        yield sample
                    else:
                        skip["cmmlu_invalid_sample"] += 1


def collect_samples(args: argparse.Namespace) -> tuple[list[MCQSample], list[MCQSample], Counter[str]]:
    skip: Counter[str] = Counter()
    train: list[MCQSample] = []
    eval_samples: list[MCQSample] = []
    cmmlu_dir = Path(args.cmmlu_dir)
    mmlu_dir = Path(args.mmlu_dir)

    if args.language_mode in {"mixed", "en"}:
        train.extend(iter_mmlu(mmlu_dir, "train", skip))
        eval_samples.extend(iter_mmlu(mmlu_dir, "eval", skip))
    if args.language_mode in {"mixed", "zh"}:
        train.extend(iter_cmmlu(cmmlu_dir, "train", skip))
        eval_samples.extend(iter_cmmlu(cmmlu_dir, "eval", skip))
    return train, eval_samples, skip


def sample_and_shuffle(samples: list[MCQSample], max_samples: int, rng: random.Random) -> list[MCQSample]:
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if max_samples > 0:
        shuffled = shuffled[:max_samples]
    return shuffled


def sft_record(sample: MCQSample, rng: random.Random) -> dict[str, str]:
    system = rng.choice(SYSTEM_TEMPLATES)
    prompt = rng.choice(PROMPT_TEMPLATES).format(body=make_body(sample))
    return {
        "system": system,
        "prompt": prompt,
        "response": f"正确答案是 {sample.answer}",
    }


def eval_record(sample: MCQSample, rng: random.Random) -> dict[str, Any]:
    system = rng.choice(SYSTEM_TEMPLATES)
    prompt = rng.choice(PROMPT_TEMPLATES).format(body=make_body(sample))
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "metadata": {
            "answer": sample.answer,
            "id": sample.sample_id,
            "source": sample.source,
            "split": sample.split,
            "subject": sample.subject,
            "language": sample.language,
        },
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def summary_path(path: Path) -> Path:
    return path.with_suffix(".summary.json")


def build_summary(
    args: argparse.Namespace,
    samples: list[MCQSample],
    skip: Counter[str],
    output_path: Path,
    kind: str,
    overlap_with_other_split: int,
) -> dict[str, Any]:
    by_source = Counter(sample.source for sample in samples)
    by_language = Counter(sample.language for sample in samples)
    by_subject = Counter(f"{sample.source}:{sample.subject or 'unknown'}" for sample in samples)
    return {
        "kind": kind,
        "output_path": str(output_path),
        "seed": args.seed,
        "language_mode": args.language_mode,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "num_samples": len(samples),
        "by_source": dict(sorted(by_source.items())),
        "by_language": dict(sorted(by_language.items())),
        "top_subjects": dict(by_subject.most_common(30)),
        "overlap_with_other_split": overlap_with_other_split,
        "skip_reasons": dict(sorted(skip.items())),
    }


def main() -> None:
    args = parse_args()
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    train_raw, eval_raw, skip = collect_samples(args)

    train_samples = sample_and_shuffle(train_raw, args.max_train_samples, train_rng)
    eval_samples = sample_and_shuffle(eval_raw, args.max_eval_samples, eval_rng)
    train_ids = {sample.sample_id for sample in train_samples}
    eval_ids = {sample.sample_id for sample in eval_samples}
    overlap = len(train_ids & eval_ids)

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    train_count = write_jsonl(train_out, (sft_record(sample, train_rng) for sample in train_samples))
    eval_count = write_jsonl(eval_out, (eval_record(sample, eval_rng) for sample in eval_samples))

    train_summary = build_summary(args, train_samples, skip, train_out, "train", overlap)
    eval_summary = build_summary(args, eval_samples, skip, eval_out, "eval", overlap)
    summary_path(train_out).write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path(eval_out).write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote train={train_count} -> {train_out}")
    print(f"[OK] wrote eval={eval_count} -> {eval_out}")
    print(f"[OK] train/eval sample_id overlap={overlap}")


if __name__ == "__main__":
    main()
