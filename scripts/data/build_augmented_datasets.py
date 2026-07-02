#!/usr/bin/env python3
"""Build auxiliary SFT datasets from raw LLM-Rec parquet files.

The generated files keep the competition-style JSONL shape:
{"system": "...", "prompt": "...", "response": "..."}

The script intentionally writes to generated_dataset/ by default and never
modifies the official dataset/ directory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq
from tqdm import tqdm


DOMAIN_PREFIX = {
    "video/video": "<|video_begin|>",
    "video/ad": "<|ad_begin|>",
    "goods": "<|prod_begin|>",
    "live": "<|living_begin|>",
}


USER_COLUMNS = [
    # video current
    "video_sampled_pid_list",
    "video_neg_feedback_list",
    "video_like_list",
    "video_comment_list",
    "video_forward_list",
    "video_collect_list",
    "video_watch_time_list",
    "video_play_done_list",
    "video_duration_list",
    "video_ts_list",
    # video history
    "video_history_sampled_pid_list",
    "video_history_neg_feedback_list",
    "video_history_like_list",
    "video_history_comment_list",
    "video_history_forward_list",
    "video_history_collect_list",
    "video_history_watch_time_list",
    "video_history_play_done_list",
    "video_history_duration_list",
    "video_history_ts_list",
    # ecommerce
    "ec_colossus_rs_item_id_list",
    "ec_colossus_rs_lagv1_list",
    "ec_colossus_rs_is_click_list",
    "ec_colossus_rs_is_cart_list",
    "ec_colossus_rs_is_buy_list",
    "ec_good_click_item_id_list_extend",
    "ec_trunc_clk_lag",
    "ec_good_order_item_id_list_extend",
    "ec_trunc_buy_lag",
    # live
    "live_hist_timestamp_list",
    "live_hist_author_id_list",
    "live_hist_play_cnt_list",
    "live_hist_valid_play_cnt_list",
    "live_hist_play_duration_list",
    "live_hist_like_cnt_list",
    "live_hist_comment_cnt_list",
    "live_hist_reduce_similar_cnt_list",
    "live_hist_report_live_cnt_list",
    "live_hist_author_category_type_list",
    "live_hist_author_type_list",
    "live_hist_follow_author_cnt_list",
    # ads
    "outer_loop_history_action_pid_list_click",
    "outer_loop_history_action_pid_list_click_ts",
    "outer_loop_history_action_pid_list_click_type",
    "outer_loop_history_action_pid_list_click_industry",
    "outer_loop_deep_target_pid",
    "outer_loop_deep_target_pid_ts",
]


SYSTEM_TEMPLATES = {
    "tag_lv3_aux": [
        "你是推荐内容语义理解助手，需要根据内容token判断其三级类目。",
        "你是一名推荐类目识别助手，负责理解SID token与内容类目的对应关系。",
        "你擅长把推荐系统中的语义ID映射到可解释的三级类目。",
        "你是内容标签分析助手，请根据给定SID识别内容所属的细分类目。",
        "你负责学习内容token和tag_lv3之间的关系，并给出准确答案。",
    ],
    "video_feedback_aux": [
        "你是用户视频兴趣分析助手，需要根据观看、互动和负反馈判断兴趣强度。",
        "你是一名视频强弱反馈识别助手，请从用户行为中区分强兴趣、弱兴趣和负反馈。",
        "你擅长挖掘用户短视频偏好，需要综合完播、点赞、评论、收藏、转发和负反馈。",
        "你是推荐系统视频行为分析助手，请根据多种反馈信号输出用户兴趣分组。",
        "你负责把视频行为序列整理成强兴趣、弱兴趣和负反馈内容。",
    ],
    "ec_funnel_aux": [
        "你是电商推荐意图分析助手，需要根据曝光、点击、加购和购买判断商品兴趣强度。",
        "你是一名商品漏斗分析助手，请理解用户从曝光到购买的转化链路。",
        "你擅长根据电商行为识别高意图商品、中意图商品和低意图商品。",
        "你是购买意图识别助手，需要按转化强度划分商品token。",
        "你负责把用户商品行为整理成可用于推荐的兴趣强度分组。",
    ],
    "time_interest_aux": [
        "你是用户兴趣时序分析助手，需要区分近期兴趣、长期兴趣和兴趣变化。",
        "你是一名近期长期偏好识别助手，请根据时间窗口判断用户兴趣重点。",
        "你擅长从行为时间线中识别短期偏好、长期偏好和兴趣漂移。",
        "你是推荐时序建模助手，请根据近期、中期和长期行为归纳兴趣。",
        "你负责把用户跨时间窗口行为转化成近期兴趣和长期兴趣。",
    ],
    "live_profile_aux": [
        "你是直播推荐画像助手，需要根据观看、互动、关注和负反馈判断用户直播偏好。",
        "你是一名直播兴趣识别助手，请根据直播互动强度输出正向和负向偏好。",
        "你擅长分析直播行为，需要识别高互动主播、普通观看主播和负反馈主播。",
        "你是直播负反馈过滤助手，请同时关注用户喜欢和不喜欢的直播内容。",
        "你负责根据直播行为和主播类型整理用户直播偏好画像。",
    ],
    "ad_intent_aux": [
        "你是广告推荐意图分析助手，需要根据点击、行业、点击类型和深度转化判断广告偏好。",
        "你是一名广告行业偏好助手，请归纳用户广告兴趣和高转化广告token。",
        "你擅长识别转化广告和普通点击广告，并总结广告行业意图。",
        "你是广告行为分析助手，请根据广告点击和深度转化输出用户广告偏好。",
        "你负责把用户广告行为整理成行业兴趣、强转化广告和点击广告。",
    ],
    "cross_domain_profile_aux": [
        "你是多域推荐用户画像助手，需要综合视频、商品、直播和广告行为生成用户兴趣画像。",
        "你是一名跨场景推荐分析助手，请根据多域强反馈归纳用户偏好和推荐目标。",
        "你擅长多域兴趣归纳，需要从视频、电商、直播和广告行为中提炼目标内容。",
        "你是综合推荐画像助手，请把不同业务域的正向行为整理成用户画像。",
        "你负责根据跨域行为摘要输出用户画像和各域推荐目标。",
    ],
}


PROMPT_TEMPLATES: dict[str, list[str]] = {
    "tag_sid_to_tag": [
        "给定内容token：{sid}。请输出该内容的三级类目。",
        "请判断下面这个SID属于哪个内容类目：{sid}",
        "根据内容token还原它的tag_lv3：{sid}",
        "内容语义ID为 {sid}，请给出对应的三级类目。",
        "请根据推荐token识别细分类目，只输出类目名称：{sid}",
    ],
    "tag_tag_to_sids": [
        "三级类目：{tag}。请从该类目下生成相关内容token，最多返回{count}个。",
        "已知内容类目为“{tag}”，请列出若干匹配的SID token。",
        "请根据tag_lv3={tag}，输出相关内容token列表。",
        "下面是一个三级类目：{tag}。请给出属于该类目的内容token。",
        "请为类目“{tag}”召回相关SID，返回JSON数组。",
    ],
    "video_feedback_aux": [
        "用户视频行为如下：\n完播或长播视频：{strong}\n互动视频：{interactive}\n普通观看视频：{weak}\n负反馈视频：{negative}\n请输出强兴趣、弱兴趣和负反馈内容。",
        "请根据以下短视频反馈划分用户兴趣强度：\n强行为：{strong}\n点赞/评论/收藏/转发：{interactive}\n弱行为：{weak}\n不感兴趣：{negative}",
        "给定用户视频行为摘要，识别强兴趣、弱兴趣、负反馈。\n强反馈内容：{strong}\n互动内容：{interactive}\n普通观看：{weak}\n负反馈：{negative}",
        "下面是一个用户的视频行为分组：\n高质量观看={strong}\n主动互动={interactive}\n一般观看={weak}\n减少推荐/负反馈={negative}\n请按JSON输出。",
        "请作为推荐特征分析助手处理视频行为：强观看 {strong}；互动 {interactive}；弱观看 {weak}；负反馈 {negative}。",
    ],
    "ec_funnel_aux": [
        "用户近期商品行为如下：\n曝光未点击：{exposed}\n点击商品：{clicked}\n加购商品：{carted}\n购买商品：{bought}\n请按兴趣强度分组输出商品token。",
        "请根据电商漏斗判断用户商品意图：曝光={exposed}；点击={clicked}；加购={carted}；购买={bought}。",
        "下面是商品转化链路摘要：\n低意图曝光：{exposed}\n中意图点击：{clicked}\n较高意图加购：{carted}\n高意图购买：{bought}\n请输出JSON。",
        "从曝光、点击、加购、购买行为中识别用户商品偏好。\n曝光未点：{exposed}\n点击：{clicked}\n加购：{carted}\n购买：{bought}",
        "请分析商品行为漏斗并划分高、中、低意图：\n{summary}",
    ],
    "time_interest_aux": [
        "用户长期稳定行为：{long_term}\n用户近7天高频行为：{mid_term}\n用户最近1天强行为：{recent}\n请输出近期兴趣、长期兴趣和可能的兴趣漂移。",
        "请按时间窗口归纳兴趣：\n更早行为：{long_term}\n中期行为：{mid_term}\n近期行为：{recent}",
        "下面给出用户行为的时间分桶，请判断近期和长期兴趣。\n长期={long_term}\n近7天={mid_term}\n近1天={recent}",
        "用户兴趣时间线摘要：长期偏好 {long_term}；中期偏好 {mid_term}；近期强偏好 {recent}。请输出JSON。",
        "请从以下时间窗口行为中识别兴趣变化：\n近期：{recent}\n中期：{mid_term}\n长期：{long_term}",
    ],
    "live_profile_aux": [
        "用户直播行为如下：\n长时间有效观看并关注：{positive}\n点赞或评论互动：{interactive}\n普通观看：{neutral}\n减少相似或举报：{negative}\n直播品类线索：{types}\n请输出直播偏好画像。",
        "请根据直播互动强度识别偏好：强正向={positive}；互动={interactive}；普通={neutral}；负反馈={negative}；主播类型={types}。",
        "下面是直播行为摘要：\n高互动主播：{positive}\n轻互动主播：{interactive}\n一般观看主播：{neutral}\n不感兴趣主播：{negative}\n类型线索：{types}",
        "根据观看、互动、关注和负反馈，输出用户直播画像。\n正向直播：{positive}\n负向直播：{negative}\n品类：{types}",
        "请分析用户直播偏好：有效观看/关注 {positive}，点赞评论 {interactive}，普通观看 {neutral}，减少相似/举报 {negative}。",
    ],
    "ad_intent_aux": [
        "用户广告行为如下：\n点击广告：{clicked}\n深度转化广告：{converted}\n广告行业线索：{industries}\n点击类型线索：{click_types}\n请输出广告行业偏好、强转化广告和普通点击广告。",
        "请根据广告点击、行业和深度转化判断用户广告意图：点击={clicked}；深转化={converted}；行业={industries}；类型={click_types}。",
        "下面是广告行为摘要：\n普通点击广告：{clicked}\n强转化广告：{converted}\n行业偏好：{industries}\n点击类型：{click_types}",
        "请归纳用户广告兴趣并返回JSON。\n广告点击：{clicked}\n广告深转化：{converted}\n行业：{industries}",
        "根据广告行为识别强转化目标和行业兴趣：深转化 {converted}；点击 {clicked}；行业 {industries}；点击类型 {click_types}。",
    ],
    "cross_domain_profile_aux": [
        "用户多域行为摘要：\n视频强兴趣：{video}\n商品高意图：{goods}\n直播正向互动：{live}\n广告深度转化：{ad}\n请输出用户兴趣画像和各域推荐目标。",
        "请综合以下跨域行为生成推荐画像：视频={video}；商品={goods}；直播={live}；广告={ad}。",
        "下面给出用户在四个场景中的强反馈内容，请归纳画像并输出各域目标。\n视频：{video}\n电商：{goods}\n直播：{live}\n广告：{ad}",
        "请从视频、商品、直播、广告强行为中提炼用户兴趣。\n视频强行为 {video}\n商品强行为 {goods}\n直播强行为 {live}\n广告强行为 {ad}",
        "跨场景行为摘要如下：视频兴趣 {video}，商品意图 {goods}，直播偏好 {live}，广告转化 {ad}。请输出JSON。",
    ],
}


@dataclass
class TaskWriter:
    name: str
    path: Path
    limit: int
    fp: Any = None
    count: int = 0
    skip_reasons: Counter = field(default_factory=Counter)

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()
            self.fp = None

    def can_write(self) -> bool:
        return self.limit <= 0 or self.count < self.limit

    def write(self, record: dict[str, str]) -> bool:
        if not self.can_write():
            self.skip_reasons["limit_reached"] += 1
            return False
        self.fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build augmented LLM-Rec SFT datasets.")
    parser.add_argument("--raw-dir", default="Explorer_LLM_Rec_Competition/data")
    parser.add_argument("--out-dir", default="generated_dataset")
    parser.add_argument("--max-rows", type=int, default=50000, help="0 means unlimited.")
    parser.add_argument("--max-samples-per-task", type=int, default=5000, help="0 means unlimited.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-sids-per-group", type=int, default=20)
    parser.add_argument("--max-cross-domain-sids", type=int, default=10)
    return parser.parse_args()


def discover_table(raw_dir: Path, table_name: str) -> list[Path]:
    if not raw_dir.exists():
        return []
    files: list[Path] = []
    for child in sorted(raw_dir.iterdir()):
        if child.is_dir() and child.name.strip() == table_name:
            files.extend(sorted(child.rglob("*.parquet")))
    return files


def sid_token(domain: str, sid_three: list[Any]) -> str | None:
    prefix = DOMAIN_PREFIX.get(domain)
    if prefix is None or sid_three is None or len(sid_three) != 3:
        return None
    try:
        a, b, c = (int(float(x)) for x in sid_three)
    except (TypeError, ValueError):
        return None
    return f"{prefix}<s_a_{a}><s_b_{b}><s_c_{c}>"


def load_sid_map(raw_dir: Path) -> dict[tuple[str, int], str]:
    sid_map: dict[tuple[str, int], str] = {}
    for path in discover_table(raw_dir, "OneReason_Pid2Sid"):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["pid", "domain", "sid_three"], batch_size=65536):
            data = batch.to_pydict()
            for pid, domain, sid_three in zip(data["pid"], data["domain"], data["sid_three"]):
                token = sid_token(str(domain), sid_three)
                if token is not None and pid is not None:
                    sid_map[(str(domain), int(pid))] = token
    return sid_map


def load_tag_rows(raw_dir: Path, sid_map: dict[tuple[str, int], str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in discover_table(raw_dir, "OneReason_Pid2Tag"):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["pid", "domain", "tag_lv3"], batch_size=65536):
            data = batch.to_pydict()
            for pid, domain, tag in zip(data["pid"], data["domain"], data["tag_lv3"]):
                if pid is None or domain is None or not tag:
                    continue
                token = sid_map.get((str(domain), int(pid)))
                if token:
                    rows.append({"domain": str(domain), "pid": str(pid), "sid": token, "tag": str(tag)})
    return rows


def choose(rng: random.Random, values: list[str]) -> str:
    return rng.choice(values)


def fmt_list(values: list[str], empty: str = "无") -> str:
    return ", ".join(values) if values else empty


def trim(values: Iterable[str], max_items: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
            if max_items > 0 and len(out) >= max_items:
                break
    return out


def response_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def make_record(rng: random.Random, task: str, prompt_key: str, kwargs: dict[str, Any], response: str) -> dict[str, str]:
    system = choose(rng, SYSTEM_TEMPLATES[task])
    prompt = choose(rng, PROMPT_TEMPLATES[prompt_key]).format(**kwargs)
    return {"system": system, "prompt": prompt, "response": response}


def build_tag_dataset(
    rows: list[dict[str, str]],
    writer: TaskWriter,
    rng: random.Random,
    max_sids_per_group: int,
) -> None:
    rng.shuffle(rows)
    by_tag: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_tag[row["tag"]].append(row["sid"])

    for row in rows:
        if not writer.can_write():
            return
        record = make_record(
            rng,
            "tag_lv3_aux",
            "tag_sid_to_tag",
            {"sid": row["sid"]},
            row["tag"],
        )
        writer.write(record)

    tag_items = list(by_tag.items())
    rng.shuffle(tag_items)
    for tag, sids in tag_items:
        if not writer.can_write():
            return
        sids = trim(sids, max_sids_per_group)
        if not sids:
            writer.skip_reasons["tag_without_sid"] += 1
            continue
        record = make_record(
            rng,
            "tag_lv3_aux",
            "tag_tag_to_sids",
            {"tag": tag, "count": len(sids)},
            response_json(sids),
        )
        writer.write(record)


def safe_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    return value if isinstance(value, list) else []


def aligned_events(row: dict[str, Any], spec: dict[str, str], domain: str, sid_map: dict[tuple[str, int], str]) -> list[dict[str, Any]]:
    pids = safe_list(row, spec["pid"])
    events = []
    for i, pid in enumerate(pids):
        if pid is None:
            continue
        token = sid_map.get((domain, int(pid)))
        if not token:
            continue
        event = {"sid": token, "idx": i}
        for name, column in spec.items():
            if name == "pid":
                continue
            values = safe_list(row, column)
            event[name] = values[i] if i < len(values) else None
        events.append(event)
    return events


def truthy(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def watch_ratio(event: dict[str, Any]) -> float:
    watch = event.get("watch_time")
    duration = event.get("duration")
    try:
        watch_f = float(watch)
        duration_f = float(duration)
    except (TypeError, ValueError):
        return 0.0
    if duration_f <= 0:
        return 0.0
    return min(watch_f / duration_f, 3.0)


def video_groups(row: dict[str, Any], sid_map: dict[tuple[str, int], str], max_items: int) -> dict[str, list[str]]:
    specs = [
        {
            "pid": "video_sampled_pid_list",
            "neg": "video_neg_feedback_list",
            "like": "video_like_list",
            "comment": "video_comment_list",
            "forward": "video_forward_list",
            "collect": "video_collect_list",
            "watch_time": "video_watch_time_list",
            "play_done": "video_play_done_list",
            "duration": "video_duration_list",
            "ts": "video_ts_list",
        },
        {
            "pid": "video_history_sampled_pid_list",
            "neg": "video_history_neg_feedback_list",
            "like": "video_history_like_list",
            "comment": "video_history_comment_list",
            "forward": "video_history_forward_list",
            "collect": "video_history_collect_list",
            "watch_time": "video_history_watch_time_list",
            "play_done": "video_history_play_done_list",
            "duration": "video_history_duration_list",
            "ts": "video_history_ts_list",
        },
    ]
    events: list[dict[str, Any]] = []
    for spec in specs:
        events.extend(aligned_events(row, spec, "video/video", sid_map))
    events.sort(key=lambda x: x.get("ts") or 0, reverse=True)

    negative = []
    interactive = []
    strong = []
    weak = []
    for event in events:
        sid = event["sid"]
        if truthy(event.get("neg")):
            negative.append(sid)
        elif any(truthy(event.get(k)) for k in ["like", "comment", "forward", "collect"]):
            interactive.append(sid)
            strong.append(sid)
        elif truthy(event.get("play_done")) or watch_ratio(event) >= 0.8:
            strong.append(sid)
        else:
            weak.append(sid)
    strong = trim(strong, max_items)
    interactive = trim(interactive, max_items)
    weak = trim(weak, max_items)
    negative = trim(negative, max_items)
    return {"strong": strong, "interactive": interactive, "weak": weak, "negative": negative}


def ecommerce_groups(row: dict[str, Any], sid_map: dict[tuple[str, int], str], max_items: int) -> dict[str, list[str]]:
    show_events = aligned_events(
        row,
        {
            "pid": "ec_colossus_rs_item_id_list",
            "click": "ec_colossus_rs_is_click_list",
            "cart": "ec_colossus_rs_is_cart_list",
            "buy": "ec_colossus_rs_is_buy_list",
            "lag": "ec_colossus_rs_lagv1_list",
        },
        "goods",
        sid_map,
    )
    show_events.sort(key=lambda x: x.get("lag") if x.get("lag") is not None else 10**9)

    clicked_extend = aligned_events(
        row,
        {"pid": "ec_good_click_item_id_list_extend", "lag": "ec_trunc_clk_lag"},
        "goods",
        sid_map,
    )
    bought_extend = aligned_events(
        row,
        {"pid": "ec_good_order_item_id_list_extend", "lag": "ec_trunc_buy_lag"},
        "goods",
        sid_map,
    )
    clicked_extend.sort(key=lambda x: x.get("lag") if x.get("lag") is not None else 10**9)
    bought_extend.sort(key=lambda x: x.get("lag") if x.get("lag") is not None else 10**9)

    exposed, clicked, carted, bought = [], [], [], []
    for event in show_events:
        sid = event["sid"]
        if truthy(event.get("buy")):
            bought.append(sid)
        elif truthy(event.get("cart")):
            carted.append(sid)
        elif truthy(event.get("click")):
            clicked.append(sid)
        else:
            exposed.append(sid)
    clicked.extend(event["sid"] for event in clicked_extend)
    bought.extend(event["sid"] for event in bought_extend)
    return {
        "exposed": trim(exposed, max_items),
        "clicked": trim(clicked, max_items),
        "carted": trim(carted, max_items),
        "bought": trim(bought, max_items),
    }


def live_groups(row: dict[str, Any], sid_map: dict[tuple[str, int], str], max_items: int) -> dict[str, list[str]]:
    events = aligned_events(
        row,
        {
            "pid": "live_hist_author_id_list",
            "play": "live_hist_play_cnt_list",
            "valid_play": "live_hist_valid_play_cnt_list",
            "duration": "live_hist_play_duration_list",
            "like": "live_hist_like_cnt_list",
            "comment": "live_hist_comment_cnt_list",
            "reduce": "live_hist_reduce_similar_cnt_list",
            "report": "live_hist_report_live_cnt_list",
            "category": "live_hist_author_category_type_list",
            "type": "live_hist_author_type_list",
            "follow": "live_hist_follow_author_cnt_list",
            "ts": "live_hist_timestamp_list",
        },
        "live",
        sid_map,
    )
    positive, interactive, neutral, negative, types = [], [], [], [], []
    for event in events:
        sid = event["sid"]
        if event.get("category"):
            types.append(str(event["category"]))
        if event.get("type"):
            types.append(str(event["type"]))
        if truthy(event.get("reduce")) or truthy(event.get("report")):
            negative.append(sid)
        elif truthy(event.get("follow")) or truthy(event.get("valid_play")) or truthy(event.get("duration")):
            positive.append(sid)
        elif truthy(event.get("like")) or truthy(event.get("comment")):
            interactive.append(sid)
        elif truthy(event.get("play")):
            neutral.append(sid)
    return {
        "positive": trim(positive, max_items),
        "interactive": trim(interactive, max_items),
        "neutral": trim(neutral, max_items),
        "negative": trim(negative, max_items),
        "types": trim(types, max_items),
    }


def ad_groups(row: dict[str, Any], sid_map: dict[tuple[str, int], str], max_items: int) -> dict[str, list[str]]:
    clicks = aligned_events(
        row,
        {
            "pid": "outer_loop_history_action_pid_list_click",
            "ts": "outer_loop_history_action_pid_list_click_ts",
            "click_type": "outer_loop_history_action_pid_list_click_type",
            "industry": "outer_loop_history_action_pid_list_click_industry",
        },
        "video/ad",
        sid_map,
    )
    converted = aligned_events(
        row,
        {"pid": "outer_loop_deep_target_pid", "ts": "outer_loop_deep_target_pid_ts"},
        "video/ad",
        sid_map,
    )
    clicks.sort(key=lambda x: x.get("ts") or 0, reverse=True)
    converted.sort(key=lambda x: x.get("ts") or 0, reverse=True)
    industries = [str(e["industry"]) for e in clicks if e.get("industry")]
    click_types = [str(e["click_type"]) for e in clicks if e.get("click_type")]
    conversion_from_clicks = [e["sid"] for e in clicks if str(e.get("click_type") or "").upper() == "EVENT_CONVERSION"]
    converted_sids = [e["sid"] for e in converted] + conversion_from_clicks
    clicked_sids = [e["sid"] for e in clicks]
    return {
        "clicked": trim(clicked_sids, max_items),
        "converted": trim(converted_sids, max_items),
        "industries": trim(industries, max_items),
        "click_types": trim(click_types, max_items),
    }


def bucket_time_groups(row: dict[str, Any], sid_map: dict[tuple[str, int], str], max_items: int) -> dict[str, list[str]]:
    timed: list[tuple[int, str]] = []
    for groups in [
        video_groups(row, sid_map, max_items * 3),
        ad_groups(row, sid_map, max_items * 3),
    ]:
        # These helpers already prioritize recency or conversion, so use rank as pseudo time.
        for idx, sid in enumerate(groups.get("strong", []) + groups.get("converted", [])):
            timed.append((10_000 - idx, sid))

    ec = ecommerce_groups(row, sid_map, max_items * 3)
    for idx, sid in enumerate(ec.get("bought", []) + ec.get("carted", [])):
        timed.append((8_000 - idx, sid))
    live = live_groups(row, sid_map, max_items * 3)
    for idx, sid in enumerate(live.get("positive", [])):
        timed.append((6_000 - idx, sid))

    timed.sort(reverse=True)
    sids = trim([sid for _, sid in timed], max_items * 3)
    if len(sids) < 3:
        return {"recent": [], "mid_term": [], "long_term": []}
    return {
        "recent": trim(sids[:max_items], max_items),
        "mid_term": trim(sids[max_items : max_items * 2], max_items),
        "long_term": trim(sids[max_items * 2 : max_items * 3], max_items),
    }


def nonempty_count(groups: dict[str, list[str]], keys: Iterable[str]) -> int:
    return sum(1 for key in keys if groups.get(key))


def write_user_task_records(
    row: dict[str, Any],
    sid_map: dict[tuple[str, int], str],
    writers: dict[str, TaskWriter],
    rng: random.Random,
    max_items: int,
    max_cross_items: int,
) -> None:
    video = video_groups(row, sid_map, max_items)
    if writers["video_feedback_aux"].can_write() and nonempty_count(video, ["strong", "interactive", "weak", "negative"]) >= 2:
        payload = {
            "strong_interest": trim(video["strong"] + video["interactive"], max_items),
            "weak_interest": video["weak"],
            "negative_feedback": video["negative"],
        }
        if payload["strong_interest"] or payload["negative_feedback"]:
            record = make_record(
                rng,
                "video_feedback_aux",
                "video_feedback_aux",
                {
                    "strong": fmt_list(video["strong"]),
                    "interactive": fmt_list(video["interactive"]),
                    "weak": fmt_list(video["weak"]),
                    "negative": fmt_list(video["negative"]),
                },
                response_json(payload),
            )
            writers["video_feedback_aux"].write(record)
        else:
            writers["video_feedback_aux"].skip_reasons["no_strong_or_negative"] += 1
    elif writers["video_feedback_aux"].can_write():
        writers["video_feedback_aux"].skip_reasons["insufficient_joined_video_groups"] += 1

    ec = ecommerce_groups(row, sid_map, max_items)
    if writers["ec_funnel_aux"].can_write() and nonempty_count(ec, ["exposed", "clicked", "carted", "bought"]) >= 2:
        payload = {
            "high_intent": trim(ec["bought"] + ec["carted"], max_items),
            "medium_intent": ec["clicked"],
            "low_or_unknown": ec["exposed"],
        }
        if payload["high_intent"] or payload["medium_intent"]:
            summary = f"曝光未点：{fmt_list(ec['exposed'])}\n点击：{fmt_list(ec['clicked'])}\n加购：{fmt_list(ec['carted'])}\n购买：{fmt_list(ec['bought'])}"
            record = make_record(
                rng,
                "ec_funnel_aux",
                "ec_funnel_aux",
                {
                    "exposed": fmt_list(ec["exposed"]),
                    "clicked": fmt_list(ec["clicked"]),
                    "carted": fmt_list(ec["carted"]),
                    "bought": fmt_list(ec["bought"]),
                    "summary": summary,
                },
                response_json(payload),
            )
            writers["ec_funnel_aux"].write(record)
        else:
            writers["ec_funnel_aux"].skip_reasons["no_medium_or_high_intent"] += 1
    elif writers["ec_funnel_aux"].can_write():
        if not any(ec.values()):
            writers["ec_funnel_aux"].skip_reasons["no_joined_goods_sid"] += 1
        else:
            writers["ec_funnel_aux"].skip_reasons["insufficient_goods_funnel_groups"] += 1

    time_groups = bucket_time_groups(row, sid_map, max_items)
    if writers["time_interest_aux"].can_write() and time_groups["recent"] and (time_groups["mid_term"] or time_groups["long_term"]):
        shift = "近期兴趣与较早行为存在差异，应优先关注近期强反馈内容。" if time_groups["long_term"] else "近期兴趣更明显，长期稳定偏好不足。"
        payload = {
            "recent_interest": time_groups["recent"],
            "long_term_interest": trim(time_groups["long_term"] + time_groups["mid_term"], max_items),
            "interest_shift": shift,
        }
        record = make_record(
            rng,
            "time_interest_aux",
            "time_interest_aux",
            {
                "recent": fmt_list(time_groups["recent"]),
                "mid_term": fmt_list(time_groups["mid_term"]),
                "long_term": fmt_list(time_groups["long_term"]),
            },
            response_json(payload),
        )
        writers["time_interest_aux"].write(record)
    elif writers["time_interest_aux"].can_write():
        writers["time_interest_aux"].skip_reasons["insufficient_time_buckets"] += 1

    live = live_groups(row, sid_map, max_items)
    if writers["live_profile_aux"].can_write() and nonempty_count(live, ["positive", "interactive", "neutral", "negative"]) >= 2:
        payload = {
            "positive_live": trim(live["positive"] + live["interactive"], max_items),
            "negative_live": live["negative"],
            "preferred_live_types": live["types"],
        }
        if payload["positive_live"] or payload["negative_live"]:
            record = make_record(
                rng,
                "live_profile_aux",
                "live_profile_aux",
                {
                    "positive": fmt_list(live["positive"]),
                    "interactive": fmt_list(live["interactive"]),
                    "neutral": fmt_list(live["neutral"]),
                    "negative": fmt_list(live["negative"]),
                    "types": fmt_list(live["types"]),
                },
                response_json(payload),
            )
            writers["live_profile_aux"].write(record)
    elif writers["live_profile_aux"].can_write():
        if not any(live[key] for key in ["positive", "interactive", "neutral", "negative"]):
            writers["live_profile_aux"].skip_reasons["no_joined_live_sid"] += 1
        else:
            writers["live_profile_aux"].skip_reasons["insufficient_live_groups"] += 1

    ad = ad_groups(row, sid_map, max_items)
    if writers["ad_intent_aux"].can_write() and (ad["clicked"] or ad["converted"]):
        payload = {
            "ad_industry_interest": ad["industries"],
            "strong_conversion_ads": ad["converted"],
            "clicked_ads": ad["clicked"],
        }
        if payload["strong_conversion_ads"] or payload["clicked_ads"]:
            record = make_record(
                rng,
                "ad_intent_aux",
                "ad_intent_aux",
                {
                    "clicked": fmt_list(ad["clicked"]),
                    "converted": fmt_list(ad["converted"]),
                    "industries": fmt_list(ad["industries"]),
                    "click_types": fmt_list(ad["click_types"]),
                },
                response_json(payload),
            )
            writers["ad_intent_aux"].write(record)
    elif writers["ad_intent_aux"].can_write():
        writers["ad_intent_aux"].skip_reasons["no_joined_ad_sid"] += 1

    if writers["cross_domain_profile_aux"].can_write():
        video_targets = trim(video["strong"] + video["interactive"], max_cross_items)
        goods_targets = trim(ec["bought"] + ec["carted"] + ec["clicked"], max_cross_items)
        live_targets = trim(live["positive"] + live["interactive"], max_cross_items)
        ad_targets = trim(ad["converted"] + ad["clicked"], max_cross_items)
        if nonempty_count(
            {
                "video": video_targets,
                "goods": goods_targets,
                "live": live_targets,
                "ad": ad_targets,
            },
            ["video", "goods", "live", "ad"],
        ) >= 2:
            payload = {
                "profile": "用户在多个业务域存在正向行为，应优先结合近期强反馈和跨域高意图内容生成推荐目标。",
                "video_targets": video_targets,
                "goods_targets": goods_targets,
                "live_targets": live_targets,
                "ad_targets": ad_targets,
            }
            record = make_record(
                rng,
                "cross_domain_profile_aux",
                "cross_domain_profile_aux",
                {
                    "video": fmt_list(video_targets),
                    "goods": fmt_list(goods_targets),
                    "live": fmt_list(live_targets),
                    "ad": fmt_list(ad_targets),
                },
                response_json(payload),
            )
            writers["cross_domain_profile_aux"].write(record)
        else:
            writers["cross_domain_profile_aux"].skip_reasons["insufficient_positive_domains"] += 1


def create_writers(out_dir: Path, limit: int) -> dict[str, TaskWriter]:
    names = [
        "tag_lv3_aux",
        "video_feedback_aux",
        "ec_funnel_aux",
        "time_interest_aux",
        "live_profile_aux",
        "ad_intent_aux",
        "cross_domain_profile_aux",
    ]
    return {name: TaskWriter(name, out_dir / f"{name}.jsonl", limit) for name in names}


def iter_user_rows(files: list[Path], columns: list[str], batch_size: int, max_rows: int):
    seen = 0
    for path in files:
        pf = pq.ParquetFile(path)
        available = [column for column in columns if column in pf.schema_arrow.names]
        if not available:
            continue
        for batch in pf.iter_batches(columns=available, batch_size=batch_size):
            data = batch.to_pydict()
            row_count = batch.num_rows
            for idx in range(row_count):
                if max_rows > 0 and seen >= max_rows:
                    return
                seen += 1
                yield {column: data[column][idx] for column in available}


def validate_record(record: dict[str, str]) -> None:
    for key in ["system", "prompt", "response"]:
        if key not in record or not isinstance(record[key], str):
            raise ValueError(f"invalid record missing string {key}: {record!r}")


def build_summary(args: argparse.Namespace, writers: dict[str, TaskWriter], sid_map: dict[tuple[str, int], str], tag_rows: list[dict[str, str]], rows_read: int) -> dict[str, Any]:
    sid_domain_counts = Counter(domain for domain, _ in sid_map.keys())
    tag_domain_counts = Counter(row["domain"] for row in tag_rows)
    return {
        "args": {
            "raw_dir": args.raw_dir,
            "out_dir": args.out_dir,
            "max_rows": args.max_rows,
            "max_samples_per_task": args.max_samples_per_task,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_sids_per_group": args.max_sids_per_group,
            "max_cross_domain_sids": args.max_cross_domain_sids,
        },
        "source_stats": {
            "sid_map_size": len(sid_map),
            "sid_map_domain_counts": dict(sorted(sid_domain_counts.items())),
            "tag_rows_with_sid": len(tag_rows),
            "tag_domain_counts": dict(sorted(tag_domain_counts.items())),
            "user_rows_read": rows_read,
        },
        "tasks": {
            name: {
                "file": str(writer.path),
                "records": writer.count,
                "skip_reasons": dict(writer.skip_reasons),
            }
            for name, writer in writers.items()
        },
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    print(f"[INFO] loading SID map from {raw_dir}", file=sys.stderr)
    sid_map = load_sid_map(raw_dir)
    print(f"[INFO] SID map entries: {len(sid_map):,}", file=sys.stderr)

    print("[INFO] loading tag rows", file=sys.stderr)
    tag_rows = load_tag_rows(raw_dir, sid_map)
    print(f"[INFO] tag rows with SID: {len(tag_rows):,}", file=sys.stderr)

    writers = create_writers(out_dir, args.max_samples_per_task)
    for writer in writers.values():
        writer.open()

    try:
        build_tag_dataset(tag_rows, writers["tag_lv3_aux"], rng, args.max_sids_per_group)

        user_files = discover_table(raw_dir, "OneReason_UserProfile")
        rows_read = 0
        for row in tqdm(
            iter_user_rows(user_files, USER_COLUMNS, args.batch_size, args.max_rows),
            desc="UserProfile rows",
            unit="row",
        ):
            rows_read += 1
            write_user_task_records(
                row,
                sid_map,
                writers,
                rng,
                args.max_sids_per_group,
                args.max_cross_domain_sids,
            )
            if all(not writer.can_write() for writer in writers.values()):
                break
    finally:
        for writer in writers.values():
            writer.close()

    # Validate generated JSONL shape quickly.
    for writer in writers.values():
        with writer.path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                if not line.strip():
                    continue
                validate_record(json.loads(line))
                if line_no >= 5:
                    break

    summary = build_summary(args, writers, sid_map, tag_rows, rows_read)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote augmented datasets to {out_dir}", file=sys.stderr)
    for name, task in summary["tasks"].items():
        print(f"[OK] {name}: {task['records']:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
