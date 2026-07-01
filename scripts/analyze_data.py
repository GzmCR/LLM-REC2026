#!/usr/bin/env python3
"""EDA utilities for the LLM-Rec competition data.

The script focuses on two data sources:
1. SFT JSONL files under dataset/.
2. Raw parquet shards under Explorer_LLM_Rec_Competition/data/.

It writes Markdown, CSV, and PNG artifacts into an output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


SID_RE = re.compile(
    r"<\|(?:video|ad|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
PREFIX_RE = re.compile(r"<\|(video|ad|prod|living)_begin\|>")


TASK_PATTERNS = [
    ("recommendation", "懂推荐"),
    ("user_behavior_mining", "懂用户"),
    ("item_understanding", "懂物料"),
]


SEQUENCE_COLUMNS = {
    "goods_purchase": ("goods", "ec_item_id_list"),
    "goods_recent_show": ("goods", "ec_colossus_rs_item_id_list"),
    "goods_click_extend": ("goods", "ec_good_click_item_id_list_extend"),
    "goods_order_extend": ("goods", "ec_good_order_item_id_list_extend"),
    "video_current": ("video/video", "video_sampled_pid_list"),
    "video_history": ("video/video", "video_history_sampled_pid_list"),
    "live_history": ("live", "live_hist_author_id_list"),
    "ad_pos": ("video/ad", "outer_loop_history_action_pid_list_pos"),
    "ad_click": ("video/ad", "outer_loop_history_action_pid_list_click"),
    "ad_deep_target": ("video/ad", "outer_loop_deep_target_pid"),
}


ALIGNMENT_GROUPS = {
    "video_sampled_pid_list": [
        "video_neg_feedback_list",
        "video_like_list",
        "video_comment_list",
        "video_forward_list",
        "video_collect_list",
        "video_watch_time_list",
        "video_play_done_list",
        "video_duration_list",
        "video_ts_list",
    ],
    "video_history_sampled_pid_list": [
        "video_history_neg_feedback_list",
        "video_history_like_list",
        "video_history_comment_list",
        "video_history_forward_list",
        "video_history_collect_list",
        "video_history_watch_time_list",
        "video_history_play_done_list",
        "video_history_duration_list",
        "video_history_ts_list",
    ],
    "outer_loop_history_action_pid_list_pos": [
        "outer_loop_history_action_pid_list_pos_ts",
    ],
    "outer_loop_history_action_pid_list_click": [
        "outer_loop_history_action_pid_list_click_ts",
        "outer_loop_history_action_pid_list_click_type",
        "outer_loop_history_action_pid_list_click_industry",
    ],
    "outer_loop_deep_target_pid": [
        "outer_loop_deep_target_pid_ts",
    ],
    "ec_item_id_list": [
        "ec_cvr_label_list",
        "ec_time_ms_list",
    ],
    "ec_good_click_item_id_list_extend": [
        "ec_trunc_clk_lag",
    ],
    "ec_good_order_item_id_list_extend": [
        "ec_trunc_buy_lag",
    ],
    "ec_colossus_rs_item_id_list": [
        "ec_colossus_rs_lagv1_list",
        "ec_colossus_rs_lagv2_list",
        "ec_colossus_rs_is_click_list",
        "ec_colossus_rs_is_cart_list",
        "ec_colossus_rs_is_buy_list",
    ],
    "live_hist_author_id_list": [
        "live_hist_timestamp_list",
        "live_hist_live_id_list",
        "live_hist_show_cnt_list",
        "live_hist_play_cnt_list",
        "live_hist_valid_play_cnt_list",
        "live_hist_play_duration_list",
        "live_hist_valid_play_duration_list",
        "live_hist_like_cnt_list",
        "live_hist_comment_cnt_list",
        "live_hist_reduce_similar_cnt_list",
        "live_hist_report_live_cnt_list",
        "live_hist_author_category_type_list",
        "live_hist_author_type_list",
        "live_hist_is_interactive_mp_live_list",
        "live_hist_is_building_live_list",
        "live_hist_is_local_life_live_list",
        "live_hist_is_detect_game_live_list",
        "live_hist_is_recruit_live_list",
        "live_hist_follow_author_cnt_list",
    ],
}


@dataclass
class RunningListStats:
    values: list[int] = field(default_factory=list)

    def add(self, value: int) -> None:
        self.values.append(int(value))

    def summary(self, prefix: str) -> dict[str, Any]:
        if not self.values:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_p50": 0,
                f"{prefix}_p90": 0,
                f"{prefix}_p95": 0,
                f"{prefix}_p99": 0,
                f"{prefix}_max": 0,
            }
        arr = np.asarray(self.values, dtype=np.int64)
        return {
            f"{prefix}_mean": round(float(arr.mean()), 2),
            f"{prefix}_p50": int(np.quantile(arr, 0.50)),
            f"{prefix}_p90": int(np.quantile(arr, 0.90)),
            f"{prefix}_p95": int(np.quantile(arr, 0.95)),
            f"{prefix}_p99": int(np.quantile(arr, 0.99)),
            f"{prefix}_max": int(arr.max()),
        }


@dataclass
class SftFileStats:
    file: str
    task_type: str
    rows: int = 0
    parsed_rows: int = 0
    parse_errors: int = 0
    list_wrapped_rows: int = 0
    dict_rows: int = 0
    unexpected_type_rows: int = 0
    missing_system: int = 0
    missing_prompt: int = 0
    missing_response: int = 0
    empty_system: int = 0
    empty_prompt: int = 0
    empty_response: int = 0
    prompt_has_think_marker: int = 0
    prompt_has_no_think_marker: int = 0
    response_has_think_tag: int = 0
    response_json_array: int = 0
    response_json_object: int = 0
    response_plain_text: int = 0
    response_empty_after_think: int = 0
    prompt_len: RunningListStats = field(default_factory=RunningListStats)
    response_len: RunningListStats = field(default_factory=RunningListStats)
    sid_count: RunningListStats = field(default_factory=RunningListStats)
    prefixes: Counter = field(default_factory=Counter)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "file": self.file,
            "task_type": self.task_type,
            "rows": self.rows,
            "parsed_rows": self.parsed_rows,
            "parse_errors": self.parse_errors,
            "list_wrapped_rows": self.list_wrapped_rows,
            "dict_rows": self.dict_rows,
            "unexpected_type_rows": self.unexpected_type_rows,
            "missing_system": self.missing_system,
            "missing_prompt": self.missing_prompt,
            "missing_response": self.missing_response,
            "empty_system": self.empty_system,
            "empty_prompt": self.empty_prompt,
            "empty_response": self.empty_response,
            "prompt_has_think_marker": self.prompt_has_think_marker,
            "prompt_has_no_think_marker": self.prompt_has_no_think_marker,
            "response_has_think_tag": self.response_has_think_tag,
            "response_json_array": self.response_json_array,
            "response_json_object": self.response_json_object,
            "response_plain_text": self.response_plain_text,
            "response_empty_after_think": self.response_empty_after_think,
        }
        row.update(self.prompt_len.summary("prompt_len"))
        row.update(self.response_len.summary("response_len"))
        row.update(self.sid_count.summary("sid_count"))
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LLM-Rec SFT and raw data.")
    parser.add_argument("--sft-dir", default="dataset", help="Directory with SFT jsonl files.")
    parser.add_argument(
        "--raw-dir",
        default=" Explorer_LLM_Rec_Competition/data",
        help="Directory with raw parquet tables.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/data_analysis",
        help="Directory to write analysis artifacts.",
    )
    parser.add_argument(
        "--max-anomalies",
        type=int,
        default=20000,
        help="Maximum SFT anomaly rows to write.",
    )
    parser.add_argument(
        "--max-join-items",
        type=int,
        default=200000,
        help="Maximum unique user behavior PIDs sampled per domain for join coverage.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Parquet batch size.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write Markdown and CSV artifacts.",
    )
    return parser.parse_args()


def task_type_from_name(path: Path) -> str:
    name = path.name
    for task_type, pattern in TASK_PATTERNS:
        if pattern in name:
            return task_type
    return "unknown"


def truncate_text(text: Any, limit: int = 180) -> str:
    if text is None:
        return ""
    value = str(text).replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[:limit] + f"...[len={len(value)}]"


def classify_response(response: str) -> str:
    text = (response or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if not text:
        return "empty_after_think"
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return "json_array"
        except Exception:
            pass
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return "json_object"
        except Exception:
            pass
    return "plain_text"


def record_anomaly(
    anomalies: list[dict[str, Any]],
    max_anomalies: int,
    path: Path,
    line_no: int,
    reason: str,
    detail: str = "",
    prompt: str = "",
    response: str = "",
) -> None:
    if len(anomalies) >= max_anomalies:
        return
    anomalies.append(
        {
            "file": path.name,
            "line_no": line_no,
            "reason": reason,
            "detail": detail,
            "prompt_preview": truncate_text(prompt),
            "response_preview": truncate_text(response),
        }
    )


def iter_sft_files(sft_dir: Path) -> list[Path]:
    if not sft_dir.exists():
        return []
    return sorted(p for p in sft_dir.rglob("*.jsonl") if p.is_file())


def analyze_sft(
    sft_dir: Path,
    out_dir: Path,
    max_anomalies: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = iter_sft_files(sft_dir)
    stats_by_file: list[SftFileStats] = []
    anomalies: list[dict[str, Any]] = []

    for path in tqdm(files, desc="SFT JSONL", unit="file"):
        stats = SftFileStats(file=path.name, task_type=task_type_from_name(path))
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                stats.rows += 1
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    stats.parse_errors += 1
                    record_anomaly(
                        anomalies,
                        max_anomalies,
                        path,
                        line_no,
                        "json_parse_error",
                        repr(exc),
                    )
                    continue

                if isinstance(obj, list):
                    stats.list_wrapped_rows += 1
                    obj = obj[0] if obj else {}
                elif isinstance(obj, dict):
                    stats.dict_rows += 1
                else:
                    stats.unexpected_type_rows += 1
                    record_anomaly(
                        anomalies,
                        max_anomalies,
                        path,
                        line_no,
                        "unexpected_json_type",
                        type(obj).__name__,
                    )
                    continue

                if not isinstance(obj, dict):
                    stats.unexpected_type_rows += 1
                    record_anomaly(
                        anomalies,
                        max_anomalies,
                        path,
                        line_no,
                        "unexpected_inner_type",
                        type(obj).__name__,
                    )
                    continue

                stats.parsed_rows += 1
                for key, attr in [
                    ("system", "missing_system"),
                    ("prompt", "missing_prompt"),
                    ("response", "missing_response"),
                ]:
                    if key not in obj:
                        setattr(stats, attr, getattr(stats, attr) + 1)

                system = obj.get("system", "") or ""
                prompt = obj.get("prompt", "") or ""
                response = obj.get("response", "") or ""

                if not system:
                    stats.empty_system += 1
                    record_anomaly(
                        anomalies,
                        max_anomalies,
                        path,
                        line_no,
                        "empty_system",
                        "",
                        prompt,
                        response,
                    )
                if not prompt:
                    stats.empty_prompt += 1
                    record_anomaly(
                        anomalies,
                        max_anomalies,
                        path,
                        line_no,
                        "empty_prompt",
                        "",
                        prompt,
                        response,
                    )
                if not response:
                    stats.empty_response += 1
                    record_anomaly(
                        anomalies,
                        max_anomalies,
                        path,
                        line_no,
                        "empty_response",
                        "",
                        prompt,
                        response,
                    )

                if "/think" in prompt:
                    stats.prompt_has_think_marker += 1
                if "/no_think" in prompt:
                    stats.prompt_has_no_think_marker += 1
                if "<think>" in response and "</think>" in response:
                    stats.response_has_think_tag += 1

                kind = classify_response(response)
                if kind == "json_array":
                    stats.response_json_array += 1
                elif kind == "json_object":
                    stats.response_json_object += 1
                elif kind == "empty_after_think":
                    stats.response_empty_after_think += 1
                    record_anomaly(
                        anomalies,
                        max_anomalies,
                        path,
                        line_no,
                        "response_empty_after_think",
                        "",
                        prompt,
                        response,
                    )
                else:
                    stats.response_plain_text += 1

                full_text = prompt + "\n" + response
                stats.prompt_len.add(len(prompt))
                stats.response_len.add(len(response))
                stats.sid_count.add(len(SID_RE.findall(full_text)))
                stats.prefixes.update(PREFIX_RE.findall(full_text))

        stats_by_file.append(stats)

    file_rows = [stats.as_row() for stats in stats_by_file]
    file_df = pd.DataFrame(file_rows)

    prefix_rows = []
    for stats in stats_by_file:
        for prefix in ["video", "ad", "prod", "living"]:
            prefix_rows.append(
                {
                    "file": stats.file,
                    "task_type": stats.task_type,
                    "prefix": prefix,
                    "count": stats.prefixes.get(prefix, 0),
                }
            )
    prefix_df = pd.DataFrame(prefix_rows)
    anomalies_df = pd.DataFrame(anomalies)

    write_csv(file_df, out_dir / "sft_file_summary.csv")
    write_csv(prefix_df, out_dir / "sft_prefix_summary.csv")
    write_csv(anomalies_df, out_dir / "sft_anomalies.csv")
    return file_df, prefix_df, anomalies_df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def normalize_table_name(path: Path) -> str:
    return path.name.strip()


def discover_parquet_tables(raw_dir: Path) -> dict[str, list[Path]]:
    tables: dict[str, list[Path]] = {}
    if not raw_dir.exists():
        return tables
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        files = sorted(p for p in child.rglob("*.parquet") if p.is_file())
        if files:
            tables[normalize_table_name(child)] = files
    return tables


def parquet_schema_names(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def analyze_raw_tables(
    raw_dir: Path,
    out_dir: Path,
    batch_size: int,
    max_join_items: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = discover_parquet_tables(raw_dir)
    table_rows = []
    schema_rows = []
    domain_rows = []

    for table_name, files in tqdm(tables.items(), desc="Raw tables", unit="table"):
        total_rows = 0
        total_row_groups = 0
        total_bytes = sum(path.stat().st_size for path in files)
        columns_seen: list[str] = []
        for path in files:
            pf = pq.ParquetFile(path)
            total_rows += pf.metadata.num_rows
            total_row_groups += pf.metadata.num_row_groups
            names = pf.schema_arrow.names
            for name in names:
                if name not in columns_seen:
                    columns_seen.append(name)
            for idx, field in enumerate(pf.schema_arrow):
                schema_rows.append(
                    {
                        "table": table_name,
                        "file": str(path),
                        "column_index": idx,
                        "column": field.name,
                        "type": str(field.type),
                    }
                )
        table_rows.append(
            {
                "table": table_name,
                "files": len(files),
                "rows": total_rows,
                "row_groups": total_row_groups,
                "size_mb": round(total_bytes / 1024 / 1024, 2),
                "columns": len(columns_seen),
                "column_names": "|".join(columns_seen),
            }
        )

        if "domain" in columns_seen:
            domain_counter: Counter = Counter()
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(
                    columns=["domain"], batch_size=batch_size
                ):
                    domain_counter.update(batch.column("domain").to_pylist())
            for domain, count in sorted(domain_counter.items(), key=lambda x: str(x[0])):
                domain_rows.append(
                    {"table": table_name, "domain": domain, "rows": int(count)}
                )

    table_df = pd.DataFrame(table_rows)
    schema_df = pd.DataFrame(schema_rows)
    domain_df = pd.DataFrame(domain_rows)

    sequence_df, alignment_df, sampled_pids = analyze_user_profile_sequences(
        tables.get("OneReason_UserProfile", []), batch_size, max_join_items
    )
    join_df = analyze_join_coverage(tables, sampled_pids, batch_size)

    write_csv(table_df, out_dir / "raw_table_summary.csv")
    write_csv(schema_df, out_dir / "raw_schema_summary.csv")
    write_csv(domain_df, out_dir / "raw_domain_summary.csv")
    write_csv(sequence_df, out_dir / "raw_sequence_summary.csv")
    write_csv(alignment_df, out_dir / "raw_alignment_summary.csv")
    write_csv(join_df, out_dir / "raw_join_coverage.csv")
    return table_df, domain_df, sequence_df, alignment_df, join_df


def list_len(value: Any) -> int:
    if value is None:
        return -1
    try:
        return len(value)
    except TypeError:
        return -1


def add_pid_samples(
    sampled_pids: dict[str, set[int]],
    domain: str,
    values: Iterable[Any],
    max_join_items: int,
) -> None:
    target = sampled_pids[domain]
    if len(target) >= max_join_items:
        return
    for value in values:
        if value is None:
            continue
        for pid in value:
            if pid is not None:
                target.add(int(pid))
                if len(target) >= max_join_items:
                    return


def analyze_user_profile_sequences(
    files: list[Path],
    batch_size: int,
    max_join_items: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[int]]]:
    sequence_stats = {
        label: {
            "domain": domain,
            "column": column,
            "rows": 0,
            "nonnull_rows": 0,
            "empty_rows": 0,
            "lengths": [],
        }
        for label, (domain, column) in SEQUENCE_COLUMNS.items()
    }
    alignment_stats: dict[tuple[str, str], dict[str, Any]] = {}
    sampled_pids: dict[str, set[int]] = defaultdict(set)

    if not files:
        return pd.DataFrame(), pd.DataFrame(), sampled_pids

    all_needed_cols = set()
    for _, column in SEQUENCE_COLUMNS.values():
        all_needed_cols.add(column)
    for primary, aligned_cols in ALIGNMENT_GROUPS.items():
        all_needed_cols.add(primary)
        all_needed_cols.update(aligned_cols)
        for aligned in aligned_cols:
            alignment_stats[(primary, aligned)] = {
                "primary_column": primary,
                "aligned_column": aligned,
                "checked_rows": 0,
                "both_null_rows": 0,
                "mismatch_rows": 0,
                "primary_only_rows": 0,
                "aligned_only_rows": 0,
                "example": "",
            }

    for path in tqdm(files, desc="UserProfile", unit="file"):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        columns = sorted(all_needed_cols & available)
        if not columns:
            continue
        for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
            batch_dict = batch.to_pydict()
            batch_rows = batch.num_rows

            for label, meta in sequence_stats.items():
                column = meta["column"]
                if column not in batch_dict:
                    continue
                values = batch_dict[column]
                meta["rows"] += batch_rows
                lengths = [list_len(value) for value in values]
                nonnull_lengths = [length for length in lengths if length >= 0]
                meta["nonnull_rows"] += len(nonnull_lengths)
                meta["empty_rows"] += sum(1 for length in nonnull_lengths if length == 0)
                meta["lengths"].extend(nonnull_lengths)
                add_pid_samples(sampled_pids, meta["domain"], values, max_join_items)

            for primary, aligned_cols in ALIGNMENT_GROUPS.items():
                if primary not in batch_dict:
                    continue
                primary_values = batch_dict[primary]
                primary_lengths = [list_len(value) for value in primary_values]
                for aligned in aligned_cols:
                    if aligned not in batch_dict:
                        continue
                    stats = alignment_stats[(primary, aligned)]
                    aligned_lengths = [list_len(value) for value in batch_dict[aligned]]
                    for row_idx, (p_len, a_len) in enumerate(
                        zip(primary_lengths, aligned_lengths)
                    ):
                        stats["checked_rows"] += 1
                        if p_len < 0 and a_len < 0:
                            stats["both_null_rows"] += 1
                        elif p_len >= 0 and a_len < 0:
                            stats["primary_only_rows"] += 1
                            stats["mismatch_rows"] += 1
                        elif p_len < 0 and a_len >= 0:
                            stats["aligned_only_rows"] += 1
                            stats["mismatch_rows"] += 1
                        elif p_len != a_len:
                            stats["mismatch_rows"] += 1

                        if stats["mismatch_rows"] and not stats["example"]:
                            stats["example"] = (
                                f"file={path.name};batch_row={row_idx};"
                                f"primary_len={p_len};aligned_len={a_len}"
                            )

    sequence_rows = []
    for label, meta in sequence_stats.items():
        lengths = meta.pop("lengths")
        arr = np.asarray(lengths, dtype=np.int64) if lengths else np.asarray([], dtype=np.int64)
        rows = int(meta["rows"])
        if len(arr):
            row = {
                "sequence": label,
                "domain": meta["domain"],
                "column": meta["column"],
                "rows_checked": rows,
                "nonnull_rows": int(meta["nonnull_rows"]),
                "coverage": round(meta["nonnull_rows"] / rows, 6) if rows else 0.0,
                "empty_rows": int(meta["empty_rows"]),
                "len_mean": round(float(arr.mean()), 2),
                "len_p50": int(np.quantile(arr, 0.50)),
                "len_p90": int(np.quantile(arr, 0.90)),
                "len_p95": int(np.quantile(arr, 0.95)),
                "len_p99": int(np.quantile(arr, 0.99)),
                "len_max": int(arr.max()),
            }
        else:
            row = {
                "sequence": label,
                "domain": meta["domain"],
                "column": meta["column"],
                "rows_checked": rows,
                "nonnull_rows": 0,
                "coverage": 0.0,
                "empty_rows": 0,
                "len_mean": 0.0,
                "len_p50": 0,
                "len_p90": 0,
                "len_p95": 0,
                "len_p99": 0,
                "len_max": 0,
            }
        sequence_rows.append(row)

    alignment_rows = []
    for stats in alignment_stats.values():
        checked_rows = stats["checked_rows"]
        alignment_rows.append(
            {
                **stats,
                "mismatch_rate": round(stats["mismatch_rows"] / checked_rows, 8)
                if checked_rows
                else 0.0,
            }
        )

    return pd.DataFrame(sequence_rows), pd.DataFrame(alignment_rows), sampled_pids


def analyze_join_coverage(
    tables: dict[str, list[Path]],
    sampled_pids: dict[str, set[int]],
    batch_size: int,
) -> pd.DataFrame:
    metadata_tables = ["OneReason_Pid2Sid", "OneReason_Pid2Caption", "OneReason_Pid2Tag"]
    rows = []
    metadata_sets: dict[str, dict[str, set[int]]] = {}

    for table in metadata_tables:
        files = tables.get(table, [])
        table_sets: dict[str, set[int]] = defaultdict(set)
        for path in files:
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            if not {"pid", "domain"}.issubset(names):
                continue
            for batch in pf.iter_batches(columns=["pid", "domain"], batch_size=batch_size):
                data = batch.to_pydict()
                for pid, domain in zip(data["pid"], data["domain"]):
                    if pid is not None and domain is not None:
                        table_sets[str(domain)].add(int(pid))
        metadata_sets[table] = table_sets

    for domain in ["video/video", "video/ad", "goods", "live"]:
        user_ids = sampled_pids.get(domain, set())
        for table in metadata_tables:
            meta_ids = metadata_sets.get(table, {}).get(domain, set())
            matched = len(user_ids & meta_ids) if user_ids and meta_ids else 0
            sampled = len(user_ids)
            rows.append(
                {
                    "domain": domain,
                    "metadata_table": table,
                    "sampled_user_unique_pids": sampled,
                    "metadata_unique_pids": len(meta_ids),
                    "matched_unique_pids": matched,
                    "coverage": round(matched / sampled, 6) if sampled else 0.0,
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    out_dir: Path,
    sft_file_df: pd.DataFrame,
    sft_prefix_df: pd.DataFrame,
    raw_sequence_df: pd.DataFrame,
) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    if not sft_file_df.empty:
        plot_df = sft_file_df.copy()
        plot_df["file_label"] = plot_df["file"].map(ascii_file_label)
        plot_df = plot_df.sort_values("rows", ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=plot_df, x="file_label", y="rows", hue="task_type", dodge=False)
        plt.xticks(rotation=45, ha="right")
        plt.title("SFT records by file")
        plt.tight_layout()
        plt.savefig(figures_dir / "sft_rows_by_file.png", dpi=180)
        plt.close()

        len_df = sft_file_df.copy()
        len_df["file_label"] = len_df["file"].map(ascii_file_label)
        len_df = len_df.sort_values("prompt_len_p95", ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=len_df, x="file_label", y="prompt_len_p95", color="#4C78A8")
        plt.xticks(rotation=45, ha="right")
        plt.title("SFT prompt length p95 by file")
        plt.tight_layout()
        plt.savefig(figures_dir / "sft_prompt_len_p95.png", dpi=180)
        plt.close()

    if not sft_prefix_df.empty:
        prefix_plot_df = sft_prefix_df.copy()
        prefix_plot_df["file_label"] = prefix_plot_df["file"].map(ascii_file_label)
        pivot = prefix_plot_df.pivot_table(
            index="file_label", columns="prefix", values="count", aggfunc="sum", fill_value=0
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
        ax.set_title("SID prefix counts by SFT file")
        ax.set_ylabel("count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "sft_prefix_counts.png", dpi=180)
        plt.close()

    if not raw_sequence_df.empty:
        seq_df = raw_sequence_df.sort_values("coverage", ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=seq_df, x="sequence", y="coverage", hue="domain", dodge=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.title("Raw UserProfile sequence coverage")
        plt.tight_layout()
        plt.savefig(figures_dir / "raw_sequence_coverage.png", dpi=180)
        plt.close()

        seq_len_df = raw_sequence_df.sort_values("len_p95", ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=seq_len_df, x="sequence", y="len_p95", hue="domain", dodge=False)
        plt.xticks(rotation=45, ha="right")
        plt.title("Raw UserProfile sequence length p95")
        plt.tight_layout()
        plt.savefig(figures_dir / "raw_sequence_len_p95.png", dpi=180)
        plt.close()


def ascii_file_label(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    if stem.startswith("懂推荐"):
        suffix = stem.replace("懂推荐", "") or "all"
        return f"rec{suffix}"
    if stem.startswith("懂用户"):
        return "user"
    if stem.startswith("懂物料part"):
        suffix = stem.replace("懂物料part", "") or "all"
        return f"item{suffix}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "file"


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"
    present = [col for col in columns if col in df.columns]
    if not present:
        return "_No matching columns._"
    return df[present].head(max_rows).to_markdown(index=False)


def scalar_int(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].sum())


def write_report(
    out_dir: Path,
    sft_file_df: pd.DataFrame,
    sft_prefix_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    raw_table_df: pd.DataFrame,
    raw_domain_df: pd.DataFrame,
    raw_sequence_df: pd.DataFrame,
    raw_alignment_df: pd.DataFrame,
    raw_join_df: pd.DataFrame,
) -> None:
    report = []
    report.append("# LLM-Rec Data Analysis Report")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append(f"- SFT rows: {scalar_int(sft_file_df, 'rows'):,}")
    report.append(f"- SFT parse errors: {scalar_int(sft_file_df, 'parse_errors'):,}")
    report.append(f"- SFT empty system rows: {scalar_int(sft_file_df, 'empty_system'):,}")
    report.append(f"- SFT anomaly rows written: {len(anomalies_df):,}")
    if not raw_table_df.empty:
        report.append(f"- Raw parquet tables: {len(raw_table_df):,}")
        report.append(f"- Raw parquet rows: {scalar_int(raw_table_df, 'rows'):,}")
    report.append("")

    report.append("## SFT JSONL Summary")
    report.append("")
    report.append(
        md_table(
            sft_file_df.sort_values("rows", ascending=False)
            if not sft_file_df.empty
            else sft_file_df,
            [
                "file",
                "task_type",
                "rows",
                "parse_errors",
                "empty_system",
                "prompt_len_p50",
                "prompt_len_p95",
                "response_len_p50",
                "response_len_p95",
                "sid_count_p50",
                "sid_count_p95",
            ],
            max_rows=30,
        )
    )
    report.append("")

    if not sft_file_df.empty:
        empty_system = sft_file_df.loc[sft_file_df["empty_system"] > 0]
        if not empty_system.empty:
            report.append("### Notable SFT Quality Signals")
            report.append("")
            for _, row in empty_system.iterrows():
                report.append(
                    f"- `{row['file']}` has {int(row['empty_system']):,} empty `system` rows."
                )
            report.append("")

    report.append("### SID Prefix Distribution")
    report.append("")
    if not sft_prefix_df.empty:
        prefix_summary = (
            sft_prefix_df.groupby(["task_type", "prefix"], as_index=False)["count"]
            .sum()
            .sort_values(["task_type", "prefix"])
        )
    else:
        prefix_summary = pd.DataFrame()
    report.append(md_table(prefix_summary, ["task_type", "prefix", "count"], max_rows=50))
    report.append("")

    report.append("## Raw Parquet Summary")
    report.append("")
    report.append(
        md_table(
            raw_table_df.sort_values("rows", ascending=False)
            if not raw_table_df.empty
            else raw_table_df,
            ["table", "files", "rows", "row_groups", "size_mb", "columns"],
            max_rows=20,
        )
    )
    report.append("")

    report.append("### Raw Domain Distribution")
    report.append("")
    report.append(md_table(raw_domain_df, ["table", "domain", "rows"], max_rows=50))
    report.append("")

    report.append("### UserProfile Sequence Coverage")
    report.append("")
    report.append(
        md_table(
            raw_sequence_df.sort_values("coverage", ascending=False)
            if not raw_sequence_df.empty
            else raw_sequence_df,
            [
                "sequence",
                "domain",
                "nonnull_rows",
                "coverage",
                "len_p50",
                "len_p95",
                "len_max",
            ],
            max_rows=30,
        )
    )
    report.append("")

    report.append("### Sequence Alignment Checks")
    report.append("")
    if not raw_alignment_df.empty:
        alignment_view = raw_alignment_df.sort_values("mismatch_rows", ascending=False)
    else:
        alignment_view = raw_alignment_df
    report.append(
        md_table(
            alignment_view,
            [
                "primary_column",
                "aligned_column",
                "checked_rows",
                "mismatch_rows",
                "mismatch_rate",
                "example",
            ],
            max_rows=30,
        )
    )
    report.append("")

    report.append("### Join Coverage")
    report.append("")
    report.append(
        md_table(
            raw_join_df,
            [
                "domain",
                "metadata_table",
                "sampled_user_unique_pids",
                "metadata_unique_pids",
                "matched_unique_pids",
                "coverage",
            ],
            max_rows=50,
        )
    )
    report.append("")

    report.append("## Generated Artifacts")
    report.append("")
    artifact_names = [
        "sft_file_summary.csv",
        "sft_prefix_summary.csv",
        "sft_anomalies.csv",
        "raw_table_summary.csv",
        "raw_schema_summary.csv",
        "raw_domain_summary.csv",
        "raw_sequence_summary.csv",
        "raw_alignment_summary.csv",
        "raw_join_coverage.csv",
        "figures/",
    ]
    for name in artifact_names:
        report.append(f"- `{name}`")
    report.append("")
    report.append(
        "_Note: raw data appears to be a partial local download, so raw coverage "
        "statistics should be interpreted as local-sample diagnostics._"
    )
    report.append("")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))

    sft_dir = Path(args.sft_dir)
    raw_dir = Path(args.raw_dir)

    print(f"[INFO] analyzing SFT JSONL under: {sft_dir}", file=sys.stderr)
    sft_file_df, sft_prefix_df, anomalies_df = analyze_sft(
        sft_dir=sft_dir,
        out_dir=out_dir,
        max_anomalies=args.max_anomalies,
    )

    print(f"[INFO] analyzing raw parquet under: {raw_dir}", file=sys.stderr)
    (
        raw_table_df,
        raw_domain_df,
        raw_sequence_df,
        raw_alignment_df,
        raw_join_df,
    ) = analyze_raw_tables(
        raw_dir=raw_dir,
        out_dir=out_dir,
        batch_size=args.batch_size,
        max_join_items=args.max_join_items,
    )

    if not args.skip_plots:
        print("[INFO] writing figures", file=sys.stderr)
        make_plots(out_dir, sft_file_df, sft_prefix_df, raw_sequence_df)

    write_report(
        out_dir=out_dir,
        sft_file_df=sft_file_df,
        sft_prefix_df=sft_prefix_df,
        anomalies_df=anomalies_df,
        raw_table_df=raw_table_df,
        raw_domain_df=raw_domain_df,
        raw_sequence_df=raw_sequence_df,
        raw_alignment_df=raw_alignment_df,
        raw_join_df=raw_join_df,
    )
    print(f"[OK] wrote analysis artifacts to: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
