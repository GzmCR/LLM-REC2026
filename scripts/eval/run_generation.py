#!/usr/bin/env python3
"""Run model generation for local proxy eval sets on a GPU server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_FILES = [
    "item_understanding",
    "recommendation",
    "user_related_items",
    "user_logic_chain",
    "general_mcq",
]

DEFAULTS = {
    "item_understanding": {"num_return_sequences": 64, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 512},
    "recommendation": {"think_sequences": 32, "no_think_sequences": 32, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048},
    "user_related_items": {"num_return_sequences": 1, "temperature": 0.2, "top_p": 0.95, "max_new_tokens": 2048},
    "user_logic_chain": {"num_return_sequences": 1, "temperature": 0.2, "top_p": 0.95, "max_new_tokens": 2048},
    "general_mcq": {"num_return_sequences": 1, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model outputs for local eval sets.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/eval/local_eval.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def load_config(path: str) -> dict[str, Any]:
    cfg = {task: dict(values) for task, values in DEFAULTS.items()}
    p = Path(path)
    if not p.exists():
        return cfg
    try:
        import yaml  # type: ignore
    except Exception:
        return cfg
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    generation = raw.get("generation") or {}
    for key, values in generation.items():
        if key == "user_interest" and isinstance(values, dict):
            cfg["user_related_items"].update(values)
            cfg["user_logic_chain"].update(values)
        elif key in cfg and isinstance(values, dict):
            cfg[key].update(values)
    return cfg


def with_mode(messages: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    updated = [dict(m) for m in messages]
    if not updated:
        return updated
    user_idx = max(i for i, m in enumerate(updated) if m.get("role") == "user")
    content = updated[user_idx].get("content", "")
    if mode == "think":
        if "/no_think" in content:
            content = content.replace("/no_think", "/think")
        elif "/think" not in content:
            content += "/think"
    else:
        if "/think" in content:
            content = content.replace("/think", "/no_think")
        elif "/no_think" not in content:
            content += "/no_think"
    updated[user_idx]["content"] = content
    return updated


def chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = ""
    for msg in messages:
        text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
    return text + "assistant: "


def generate_texts(model, tokenizer, messages: list[dict[str, str]], params: dict[str, Any]) -> list[str]:
    import torch

    prompt = chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_return_sequences = int(params.get("num_return_sequences", 1))
    temperature = float(params.get("temperature", 0.0))
    top_p = float(params.get("top_p", 1.0))
    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": int(params.get("max_new_tokens", 512)),
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
    )
    model.eval()
    return model, tokenizer


def generate_for_record(model, tokenizer, rec: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    task = rec["task"]
    messages = rec["messages"]
    if task == "recommendation":
        params = dict(cfg["recommendation"])
        outputs = []
        think_n = int(params.pop("think_sequences", 32))
        no_think_n = int(params.pop("no_think_sequences", 32))
        outputs.extend(generate_texts(model, tokenizer, with_mode(messages, "think"), {**params, "num_return_sequences": think_n}))
        outputs.extend(generate_texts(model, tokenizer, with_mode(messages, "no_think"), {**params, "num_return_sequences": no_think_n}))
        return outputs
    if task in {"user_related_items", "user_logic_chain"}:
        params = cfg[task]
    else:
        params = cfg[task]
    return generate_texts(model, tokenizer, messages, params)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model, tokenizer = load_model(args)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in TASK_FILES:
        records = list(read_jsonl(eval_dir / f"{task}.jsonl") or [])
        if args.limit > 0:
            records = records[: args.limit]
        out_path = out_dir / f"{task}.jsonl"
        with out_path.open("w", encoding="utf-8") as fp:
            for idx, rec in enumerate(records, 1):
                outputs = generate_for_record(model, tokenizer, rec, cfg)
                fp.write(json.dumps({"id": rec["id"], "task": rec["task"], "outputs": outputs}, ensure_ascii=False) + "\n")
                print(f"[{task}] {idx}/{len(records)} {rec['id']}")


if __name__ == "__main__":
    main()
