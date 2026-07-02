# LLM-REC2026

快手探索者 LLM-Rec 挑战赛项目整理与数据分析工作区。本项目围绕官方提供的 OneReason / LLM-Rec 数据进行理解、质检、数据构造策略设计和后续 SFT 实验准备。

> 注意：本仓库不提交比赛原始大数据文件。尤其是 `OneReason_UserProfile` parquet 分片体积很大，请按本文说明在本地放置。

## 项目结构

```text
.
├── README.md
├── docs/
│   ├── competition_intro.md
│   ├── faq.md
│   ├── dataset_construction_strategy.md
│   └── assets/
├── scripts/
│   ├── analyze_data.py                  # 兼容入口
│   ├── build_augmented_datasets.py      # 兼容入口
│   ├── data/                            # 数据分析和增强数据构造实现
│   └── train/                           # 服务器训练脚本
├── configs/train/                       # LLaMA-Factory 训练配置
├── envs/                                # conda 环境定义
├── ./Explorer_LLM_Rec_Competition/      # 官方说明、demo 和本地原始数据
├── dataset/                             # 官方 SFT JSONL，本地目录，不提交
├── generated_dataset/                   # 增强 SFT JSONL，本地生成，不提交
├── outputs/                             # 数据分析产物，不提交
├── train_data/                          # LLaMA-Factory 训练 JSONL，不提交
└── train_output/                        # 训练 checkpoint 和日志，不提交
```

核心入口：

- `scripts/analyze_data.py`：数据分析脚本，分析 SFT JSONL 与原始 parquet 的字段覆盖、长度分布、SID 前缀、序列覆盖率和 join 覆盖率。
- `scripts/build_augmented_datasets.py`：增强数据集构造脚本，从原始 parquet 生成多模板辅助 SFT JSONL。
- `docs/dataset_construction_strategy.md`：对 `dataset/` 来源、未充分利用字段、以及后续增强数据集构造方案的整理。
- `docs/competition_intro.md`、`docs/faq.md`：比赛背景和 FAQ 笔记。
- `Explorer_LLM_Rec_Competition/README.md`：官方原始数据字段说明。
- `Explorer_LLM_Rec_Competition/demo/`：官方 demo 转换与训练流程示例。

## 数据说明

本地使用了两类数据：

1. 官方 SFT JSONL 数据，放在：

   ```text
   dataset/
   ```

   当前本地包含 `懂推荐`、`懂用户`、`懂物料` 三类任务数据。

2. 原始 parquet 数据，放在：

   ```text
   ./Explorer_LLM_Rec_Competition/data/
   ```

   主要表包括：

   - `OneReason_UserProfile`：用户多域行为序列。
   - `OneReason_Pid2Sid`：`(domain, pid)` 到 `[s_a, s_b, s_c]` 语义 ID 的映射。
   - `OneReason_Pid2Caption`：物料描述。
   - `OneReason_Pid2Tag`：三级类目标签。
   - `OneReason_General`：通用 message/reasoning 数据。

### 不提交到 GitHub 的数据

以下内容被 `.gitignore` 排除：

- `dataset/`
- `Explorer_LLM_Rec_Competition/data/`
- `generated_dataset/`
- `train_data/`
- `train_output/`
- `third_party/`
- `outputs/`
- Python cache、`.DS_Store` 等本地文件

原因：

- `OneReason_UserProfile` 本地分片约 813MB，不适合普通 GitHub 仓库。
- `dataset/懂推荐*.jsonl`、`dataset/懂用户.jsonl` 单文件较大，也容易超过 GitHub 普通仓库限制。
- `generated_dataset/`、`train_data/`、`train_output/` 属于可再生成的实验产物。
- `third_party/LLaMA-Factory/` 会在服务器安装脚本中自动克隆，不需要随仓库提交。
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
  --raw-dir "Explorer_LLM_Rec_Competition/data" \
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
docs/dataset_construction_strategy.md
```

## 后续实验建议

优先不要直接改官方 `dataset/`，建议新增辅助数据目录，例如：

```text
generated_dataset/
```

可以用下面的脚本从原始 parquet 构造辅助数据：

```bash
python scripts/build_augmented_datasets.py \
  --raw-dir "Explorer_LLM_Rec_Competition/data" \
  --out-dir generated_dataset \
  --max-rows 50000 \
  --max-samples-per-task 5000 \
  --seed 2026
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

辅助数据仍然是 `{system, prompt, response}` JSONL，可继续用官方转换脚本转成 Alpaca 格式：

```bash
python "Explorer_LLM_Rec_Competition/demo/convert_jsonl.py" \
  --input generated_dataset \
  --output /tmp/data_final_augmented.jsonl \
  --shuffle \
  --shuffle-seed 2026
```

## 服务器训练环境

本仓库提供了一套面向 GPU 服务器的 LLaMA-Factory 全量 SFT 脚本。本地没有 GPU 时不需要运行训练脚本，只需把代码和数据放到服务器后执行。

### 1. 创建 conda 环境

```bash
bash scripts/train/00_create_conda_env.sh
```

默认环境名：

```text
llm-rec2026
```

环境定义见：

```text
envs/llm-rec2026.yml
```

### 2. 安装 LLaMA-Factory 和加速依赖

```bash
bash scripts/train/01_install_llamafactory.sh
```

默认安装：

- LLaMA-Factory：目标版本 `0.9.6.dev0`
- PyTorch：`2.7.1+cu126`
- FlashAttention：`2.7.4.post1`
- Liger Kernel：`0.8.0`
- TensorBoard

如服务器 CUDA / PyTorch 版本不同，可以用环境变量覆盖，例如：

```bash
TORCH_VERSION="2.7.1+cu126" \
TORCHVISION_VERSION="0.22.1+cu126" \
TORCHAUDIO_VERSION="2.7.1+cu126" \
bash scripts/train/01_install_llamafactory.sh
```

### 3. 准备训练数据

官方数据放在：

```text
dataset/
```

如果已经生成增强数据，放在：

```text
generated_dataset/
```

然后执行：

```bash
bash scripts/train/10_prepare_dataset.sh
conda run -n llm-rec2026 python scripts/train/11_register_dataset.py
```

脚本会生成：

```text
train_data/data_official.jsonl
train_data/data_augmented.jsonl
train_data/data_mixed.jsonl
```

其中 `data_mixed.jsonl` 是训练配置默认使用的数据集。

### 4. 启动全量 SFT

训练配置：

```text
configs/train/onereason_full_sft.yaml
```

启动：

```bash
bash scripts/train/20_train_full_sft.sh
```

默认模型：

```text
OpenOneRec/OneReason-0.8B-pretrain-competition
```

默认开启：

- full SFT
- bf16 / pure_bf16
- packing + neat_packing
- FlashAttention-2
- Liger Kernel

输出目录：

```text
train_output/onereason_0.8b_full_sft/
```

### 5. 一键准备但不训练

```bash
bash scripts/train/run_all_train_setup.sh
```

这个脚本只做环境安装、数据转换和数据注册，不会启动训练。

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
