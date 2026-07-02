
# Explorer_LLM_Rec_Competition

This dataset contains user historical behaviors and content metadata, constructed from real user interaction histories. It covers behavior sequences across multiple domains for a single user and supports cross-domain recommendation, semantic-ID retrieval / generation, content understanding, and related tasks.

## Files

| File | Description |
|------|-------------|
| `OneReason_UserProfile/` | Per-user multi-domain behavior records (~500k rows) |
| `OneReason_Pid2Sid/` | Mapping from content PID to three-segment semantic ID |
| `OneReason_Pid2Caption/` | Mapping from content PID to text caption |
| `OneReason_Pid2Tag/` | Mapping from content PID to level-3 category tag |
| `OneReason_General/` | General knowledge data |

---

## Anonymization & Join Rules

- All `pid` / item id / author id / live id are hashed `int64` values.
- The user table contains no `uid`; each row is an anonymous user sample.
- Hashed item ids join the three item-metadata tables via `(domain, pid)`.
- Regular videos and ads both come from the video system, distinguished by `domain`:
  - Regular video: `video/video`
  - Ad: `video/ad`
  - Some users' video-watch sequences also contain ads.

## Domain Values

| Domain | Meaning | Associated user fields | Token prefix |
|--------|---------|------------------------|--------------|
| `video/video` | Regular short videos | `video_*`, `video_history_*` | `<\|video_begin\|>` |
| `video/ad` | Ads / outer-loop content | `outer_loop_*` | `<\|ad_begin\|>` |
| `goods` | E-commerce products | `ec_*` | `<\|prod_begin\|>` |
| `live` | Live-streaming hosts | `live_hist_author_id_list` | `<\|living_begin\|>` |

---

## OneReason_Pid2Sid

Semantic-ID mapping table. Converts a content ID into a three-segment semantic ID.

| Field | Type | Description |
|-------|------|-------------|
| `pid` | int64 | Hashed content / product / host ID |
| `domain` | string | Domain of the content (see Domain Values) |
| `sid_three` | list[float] | Three-segment semantic ID `[s_a, s_b, s_c]` |

Token concatenation example:

```text
video/video: <|video_begin|><s_a_123><s_b_456><s_c_789>
video/ad:    <|ad_begin|><s_a_123><s_b_456><s_c_789>
goods:       <|prod_begin|><s_a_123><s_b_456><s_c_789>
live:        <|living_begin|><s_a_123><s_b_456><s_c_789>
```

## OneReason_Pid2Caption

Lookup of a content's text description by hashed PID.

| Field | Type | Description |
|-------|------|-------------|
| `pid` | int64 | Hashed content / product / host ID |
| `domain` | string | Domain of the content (see Domain Values) |
| `caption` | string | Content description, or stringified list of description tags |

## OneReason_Pid2Tag

Level-3 category tag for video, ad, and live content.

| Field | Type | Description |
|-------|------|-------------|
| `pid` | int64 | Hashed content / host ID |
| `domain` | string | Domain; currently `video/video`, `video/ad`, `live` |
| `tag_lv3` | string | Level-3 category tag |

---

## OneReason_UserProfile

Each row is an anonymous user-profile sample containing behavior sequences from e-commerce, short-video, live-streaming, and ad domains.

### Sequence Alignment Rules

`user_history` fields are grouped into 4 domains. Each domain has one or more **primary sequences (pid / ID sequences)**, and every primary sequence is paired with a set of **aligned feature sequences** of identical length. If both a primary sequence and its aligned feature sequences are `null`, the user has no behavior in that domain.

#### `video/video` (short video)

| Primary sequence | Aligned feature sequences |
|------------------|---------------------------|
| `video_sampled_pid_list` | `video_neg_feedback_list`, `video_like_list`, `video_comment_list`, `video_forward_list`, `video_collect_list`, `video_watch_time_list`, `video_play_done_list`, `video_duration_list`, `video_ts_list` |
| `video_history_sampled_pid_list` | `video_history_neg_feedback_list`, `video_history_like_list`, `video_history_comment_list`, `video_history_forward_list`, `video_history_collect_list`, `video_history_watch_time_list`, `video_history_play_done_list`, `video_history_duration_list`, `video_history_ts_list` |

#### `video/ad` (ads / outer-loop)

| Primary sequence | Aligned feature sequences |
|------------------|---------------------------|
| `outer_loop_history_action_pid_list_pos` | `outer_loop_history_action_pid_list_pos_ts` |
| `outer_loop_history_action_pid_list_click` | `outer_loop_history_action_pid_list_click_ts`, `outer_loop_history_action_pid_list_click_type`, `outer_loop_history_action_pid_list_click_industry` |
| `outer_loop_deep_target_pid` | `outer_loop_deep_target_pid_ts` |

#### `goods` (e-commerce)

| Primary sequence | Aligned feature sequences |
|------------------|---------------------------|
| `ec_item_id_list` | `ec_cvr_label_list`, `ec_time_ms_list` |
| `ec_good_click_item_id_list_extend` | `ec_trunc_clk_lag` |
| `ec_good_order_item_id_list_extend` | `ec_trunc_buy_lag` |
| `ec_colossus_rs_item_id_list` | `ec_colossus_rs_lagv1_list`, `ec_colossus_rs_lagv2_list`, `ec_colossus_rs_is_click_list`, `ec_colossus_rs_is_cart_list`, `ec_colossus_rs_is_buy_list` |

#### `live` (live-streaming)

| Primary sequence | Aligned feature sequences |
|------------------|---------------------------|
| `live_hist_author_id_list` | `live_hist_timestamp_list`, `live_hist_live_id_list`, `live_hist_show_cnt_list`, `live_hist_play_cnt_list`, `live_hist_valid_play_cnt_list`, `live_hist_play_duration_list`, `live_hist_valid_play_duration_list`, `live_hist_like_cnt_list`, `live_hist_comment_cnt_list`, `live_hist_reduce_similar_cnt_list`, `live_hist_report_live_cnt_list`, `live_hist_author_category_type_list`, `live_hist_author_type_list`, `live_hist_is_interactive_mp_live_list`, `live_hist_is_building_live_list`, `live_hist_is_local_life_live_list`, `live_hist_is_detect_game_live_list`, `live_hist_is_recruit_live_list`, `live_hist_follow_author_cnt_list` |

---

### E-commerce Fields

| Field | Type | Description |
|-------|------|-------------|
| `ec_item_id_list` | list[int64] | E-commerce item ID sequence |
| `ec_cvr_label_list` | list[int64] | Conversion label aligned with `ec_item_id_list` |
| `ec_time_ms_list` | list[int64] | Absolute timestamp (ms) aligned with `ec_item_id_list` |
| `ec_colossus_rs_item_id_list` | list[int64] | Recently shown e-commerce item ID sequence |
| `ec_colossus_rs_lagv1_list` | list[int64] | Hour-level lag of `ec_colossus_rs_item_id_list` relative to `ec_time_ms` |
| `ec_colossus_rs_lagv2_list` | list[int64] | Supplementary hour-level lag of `ec_colossus_rs_item_id_list` relative to `ec_time_ms` |
| `ec_colossus_rs_is_click_list` | list[int64] | Click flag for each recently shown item (0/1) |
| `ec_colossus_rs_is_cart_list` | list[int64] | Add-to-cart flag for each recently shown item (0/1) |
| `ec_colossus_rs_is_buy_list` | list[int64] | Purchase flag for each recently shown item (0/1) |
| `ec_good_click_item_id_list_extend` | list[int64] | Recently clicked e-commerce item ID sequence |
| `ec_trunc_clk_lag` | list[int64] | Hour-level lag of `ec_good_click_item_id_list_extend` relative to `ec_time_ms` |
| `ec_good_order_item_id_list_extend` | list[int64] | Recently purchased e-commerce item ID sequence |
| `ec_trunc_buy_lag` | list[int64] | Hour-level lag of `ec_good_order_item_id_list_extend` relative to `ec_time_ms` |
| `ec_time_ms` | int64 | Snapshot time when e-commerce features were captured |

### Short Video Fields

| Field | Type | Description |
|-------|------|-------------|
| `video_sampled_pid_list` | list[int64] | Currently clicked short-video ID sequence |
| `video_neg_feedback_list` | list[float] | Negative-feedback flag for currently clicked videos (0/1) |
| `video_like_list` | list[float] | Like flag for currently clicked videos (0/1) |
| `video_comment_list` | list[float] | Comment flag for currently clicked videos (0/1) |
| `video_forward_list` | list[float] | Forward / share flag for currently clicked videos (0/1) |
| `video_collect_list` | list[float] | Collect flag for currently clicked videos (0/1) |
| `video_watch_time_list` | list[float] | Watch time for currently clicked videos |
| `video_play_done_list` | list[float] | Play-done (full-watch) flag for currently clicked videos (0/1) |
| `video_duration_list` | list[float] | Duration of currently clicked videos |
| `video_ts_list` | list[int64] | Absolute timestamp of currently clicked videos |
| `video_history_sampled_pid_list` | list[int64] | Historical shown short-video ID sequence |
| `video_history_neg_feedback_list` | list[float] | Negative-feedback flag for historical videos (0/1) |
| `video_history_like_list` | list[float] | Like flag for historical videos (0/1) |
| `video_history_comment_list` | list[float] | Comment flag for historical videos (0/1) |
| `video_history_forward_list` | list[float] | Forward / share flag for historical videos (0/1) |
| `video_history_collect_list` | list[float] | Collect flag for historical videos (0/1) |
| `video_history_watch_time_list` | list[float] | Watch time for historical videos |
| `video_history_play_done_list` | list[float] | Play-done flag for historical videos (0/1) |
| `video_history_duration_list` | list[float] | Duration of historical videos |
| `video_history_ts_list` | list[int64] | Absolute timestamp of historical videos |
| `video_history_seq_len` | int64 | Length of the historical-video sequence |

### Live-streaming Fields

| Field | Type | Description |
|-------|------|-------------|
| `live_hist_timestamp_list` | list[string] | Date of watched live streams|
| `live_hist_author_id_list` | list[int64] | Host ID of watched live streams |
| `live_hist_live_id_list` | list[int64] | Live-room ID of watched live streams |
| `live_hist_show_cnt_list` | list[int64] | Impression count |
| `live_hist_play_cnt_list` | list[int64] | Play count |
| `live_hist_valid_play_cnt_list` | list[int64] | Valid-play count |
| `live_hist_play_duration_list` | list[int64] | Total play duration |
| `live_hist_valid_play_duration_list` | list[int64] | Valid-play duration |
| `live_hist_like_cnt_list` | list[int64] | Like count |
| `live_hist_comment_cnt_list` | list[int64] | Comment count |
| `live_hist_reduce_similar_cnt_list` | list[int64] | "Reduce similar" / "not interested" count |
| `live_hist_report_live_cnt_list` | list[int64] | Report count |
| `live_hist_author_category_type_list` | list[string] | Host category type |
| `live_hist_author_type_list` | list[string] | Host type |
| `live_hist_is_interactive_mp_live_list` | list[int64] | Interactive mini-program live flag (0/1) |
| `live_hist_is_building_live_list` | list[int64] | Real-estate live flag (0/1) |
| `live_hist_is_local_life_live_list` | list[int64] | Local-life live flag (0/1) |
| `live_hist_is_detect_game_live_list` | list[int64] | Gaming live flag (0/1) |
| `live_hist_is_recruit_live_list` | list[int64] | Recruitment live flag (0/1) |
| `live_hist_follow_author_cnt_list` | list[int64] | Follow flag for each host (0/1) |

### Ad / Outer-loop Fields

| Field | Type | Description |
|-------|------|-------------|
| `outer_loop_history_action_pid_list_pos` | list[int64] | Recently deep-converted outer-loop ad ID sequence |
| `outer_loop_history_action_pid_list_pos_ts` | list[int64] | Absolute timestamp aligned with `outer_loop_history_action_pid_list_pos` |
| `outer_loop_history_action_pid_list_click` | list[int64] | Recently clicked outer-loop ad ID sequence |
| `outer_loop_history_action_pid_list_click_ts` | list[int64] | Absolute timestamp aligned with `outer_loop_history_action_pid_list_click` |
| `outer_loop_history_action_pid_list_click_type` | list[string] | Click type aligned with `outer_loop_history_action_pid_list_click` |
| `outer_loop_history_action_pid_list_click_industry` | list[string] | Industry aligned with `outer_loop_history_action_pid_list_click` |
| `outer_loop_deep_target_pid` | list[int64] | Deep-conversion target ID sequence of recent outer-loop ads |
| `outer_loop_deep_target_pid_ts` | list[int64] | Absolute timestamp aligned with `outer_loop_deep_target_pid` |
