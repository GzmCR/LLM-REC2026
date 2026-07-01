# LLM-REC2026

快手探索者 LLM-Rec 挑战赛项目整理与数据分析工作区。本项目围绕官方提供的 OneReason / LLM-Rec 数据进行理解、质检、数据构造策略设计和后续 SFT 实验准备。

> 注意：本仓库不提交比赛原始大数据文件。尤其是 `OneReason_UserProfile` parquet 分片体积很大，请按本文说明在本地放置。

## 项目内容

- `scripts/analyze_data.py`：数据分析脚本，分析 SFT JSONL 与原始 parquet 的字段覆盖、长度分布、SID 前缀、序列覆盖率和 join 覆盖率。
- `数据集构造与增强策略.md`：对 `dataset/` 来源、未充分利用字段、以及后续增强数据集构造方案的整理。
- `比赛介绍.md`、`常见问题.md`：比赛背景和 FAQ 笔记。
- ` Explorer_LLM_Rec_Competition/README.md`：官方原始数据字段说明。
- ` Explorer_LLM_Rec_Competition/demo/`：官方 demo 转换与训练流程示例。

## 数据说明

本地使用了两类数据：

1. 官方 SFT JSONL 数据，放在：

   ```text
   dataset/
   ```

   当前本地包含 `懂推荐`、`懂用户`、`懂物料` 三类任务数据。

2. 原始 parquet 数据，放在：

   ```text
    Explorer_LLM_Rec_Competition/data/
   ```

   目录名前面有一个空格，这是当前本地解压后的真实路径。主要表包括：

   - `OneReason_UserProfile`：用户多域行为序列。
   - `OneReason_Pid2Sid`：`(domain, pid)` 到 `[s_a, s_b, s_c]` 语义 ID 的映射。
   - `OneReason_Pid2Caption`：物料描述。
   - `OneReason_Pid2Tag`：三级类目标签。
   - `OneReason_General`：通用 message/reasoning 数据。

### 不提交到 GitHub 的数据

以下内容被 `.gitignore` 排除：

- `dataset/`
- ` Explorer_LLM_Rec_Competition/data/`
- `outputs/`
- Python cache、`.DS_Store` 等本地文件

原因：

- `OneReason_UserProfile` 本地分片约 813MB，不适合普通 GitHub 仓库。
- `dataset/懂推荐*.jsonl`、`dataset/懂用户.jsonl` 单文件较大，也容易超过 GitHub 普通仓库限制。
- 数据集应通过官方渠道下载，并在本地按上述目录结构放置。

## 环境依赖

当前分析脚本使用 Python 3，依赖：

```bash
pip install pandas pyarrow numpy matplotlib seaborn tqdm
```

如果在 Codex/本地比赛环境中运行，通常这些依赖已经存在。

## 运行数据分析

在项目根目录执行：

```bash
python scripts/analyze_data.py \
  --sft-dir dataset \
  --raw-dir " Explorer_LLM_Rec_Competition/data" \
  --out-dir outputs/data_analysis
```

输出包括：

```text
outputs/data_analysis/report.md
outputs/data_analysis/sft_file_summary.csv
outputs/data_analysis/sft_prefix_summary.csv
outputs/data_analysis/raw_table_summary.csv
outputs/data_analysis/raw_sequence_summary.csv
outputs/data_analysis/raw_join_coverage.csv
outputs/data_analysis/figures/*.png
```

当前本地分析结果要点：

- SFT JSONL 总计约 `32,480` 条。
- `懂用户.jsonl` 的 `system` 字段为空。
- 本地 `OneReason_UserProfile` 样本为 `50,000` 行。
- 原始用户行为中仍有大量强反馈、时间、类目、漏斗和广告行业字段可以用于构造增强数据。

## 数据构造理解

当前仓库没有从 `OneReason_UserProfile` 直接生成 `dataset/懂*.jsonl` 的完整官方脚本。根据样本和字段对应关系，`dataset` 大致可以理解为：

```text
UserProfile 多域行为序列
  -> 按 (domain, pid) join OneReason_Pid2Sid
  -> 转成 <|video_begin|><s_a_x><s_b_y><s_c_z> 形式
  -> 用行为标签、时间线、物料描述和模板/LLM 生成 SFT JSONL
```

当前 `dataset` 包含三类能力：

- `懂推荐`：用户多域历史行为到推荐目标 SID。
- `懂用户`：主题相关行为抽取和行为逻辑链。
- `懂物料`：物料描述与 SID token 的双向映射。

更详细的数据构造和增强建议见：

```text
数据集构造与增强策略.md
```

## 后续实验建议

优先不要直接改官方 `dataset/`，建议新增辅助数据目录，例如：

```text
generated_dataset/
```

可优先构造：

- `tag_lv3` 类目理解数据。
- 视频强反馈数据。
- 电商曝光、点击、加购、购买漏斗数据。
- 近期兴趣、长期兴趣、兴趣漂移数据。
- 直播互动画像数据。
- 广告行业和深度转化意图数据。

训练时建议先小比例混入辅助数据做消融，例如：

```text
原始 dataset: 80% - 90%
辅助 dataset: 10% - 20%
```

重点观察 SID 输出格式稳定性、无效 token 比例、强反馈理解能力和线上/本地 proxy 指标变化。

## GitHub 上传注意事项

请不要提交大数据文件。如果需要在新机器复现：

1. 克隆本仓库。
2. 从官方渠道下载 `dataset/` 和原始 parquet。
3. 按本文的目录结构放到本地。
4. 运行 `scripts/analyze_data.py` 重新生成 `outputs/`。

如果需要使用 SSH 推送 GitHub，请先确认：

```bash
ssh -T git@github.com
```

如果提示没有权限，需要生成并添加 SSH key：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```

然后把输出的公钥添加到 GitHub：

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

再测试：

```bash
ssh -T git@github.com
```

成功后即可推送。
