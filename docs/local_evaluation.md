# 本地评测复现说明

本目录下的本地评测是官方评测的 proxy，用于比较不同训练方案的相对变化，不承诺复现官方隐藏集绝对分数。

官方 base model 参考分数：

```text
总分：0.6655
懂物料：0.1533
懂用户：0.0000, 0.0055
懂推荐：0.0864, 0.0544, 0.1372, 0.0900
懂世界：0.1387
```

## 任务对应

- `懂物料`：物料描述生成 itemic pattern，本地用 `Pass@64`。
- `懂推荐`：用户历史生成推荐 itemic pattern，本地按域计算 `Pass@64`。
- `懂用户`：分为相关行为筛选 F1 和兴趣演化链规则 proxy。
- `懂世界`：官方样例是单选/多选题，本地用完全匹配 Accuracy。

## 构造验证集

如果已经下载 CMMLU/MMLU，先构造 `懂世界` 本地验证池：

```bash
python scripts/build_general_mcq_dataset.py \
  --cmmlu-dir dataset/CMMLU \
  --mmlu-dir dataset/MMLU \
  --train-out generated_dataset/general_mcq_aux.jsonl \
  --eval-out data_eval/general_mcq.jsonl \
  --seed 2026 \
  --max-train-samples 0 \
  --max-eval-samples 0
```

默认切分：

- 训练增强集：`MMLU auxiliary_train + CMMLU test`。
- 验证池：`MMLU validation + CMMLU dev`。
- `MMLU test` 暂时保留，不混入训练或验证。

生成的 `generated_dataset/general_mcq_aux.jsonl` 是 `{system,prompt,response}` 格式，会被训练准备脚本自动混入；`data_eval/general_mcq.jsonl` 是 `messages + metadata.answer` 格式，用于本地评测。

然后从官方 SFT 和常识问答验证池构造固定 eval set：

```bash
python scripts/eval/build_eval_sets.py \
  --sft-dir dataset \
  --mcq-path data_eval/general_mcq.jsonl \
  --out-dir outputs/eval/eval_sets \
  --seed 2026 \
  --eval-ratio 0.08 \
  --max-per-task 1000
```

`data_eval/general_mcq.jsonl` 不提交 Git。它应兼容官方样例格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个非常聪明的助手，请直接遵循指示作答。"},
    {"role": "user", "content": "请回答以下问题：...请按以下格式作答：\"正确答案是 ...\""}
  ],
  "metadata": {"answer": "ABC"}
}
```

## 生成预测

真实模型生成需要 GPU 服务器：

```bash
python scripts/eval/run_generation.py \
  --model-path train_output/onereason_0.8b_full_sft \
  --eval-dir outputs/eval/eval_sets \
  --out-dir outputs/eval/predictions \
  --config configs/eval/local_eval.yaml
```

本地无 GPU 时可以生成 mock 预测测试 scorer：

```bash
python scripts/eval/mock_predictions.py \
  --eval-dir outputs/eval/eval_sets \
  --out-dir outputs/eval/mock_predictions
```

## 打分

```bash
python scripts/eval/score_eval.py \
  --eval-dir outputs/eval/eval_sets \
  --pred-dir outputs/eval/predictions \
  --out-dir outputs/eval/reports
```

输出包括：

```text
outputs/eval/reports/summary.json
outputs/eval/reports/summary.md
outputs/eval/reports/item_understanding.csv
outputs/eval/reports/recommendation.csv
outputs/eval/reports/user_interest_related.csv
outputs/eval/reports/user_interest_logic_chain.csv
outputs/eval/reports/general_mcq.csv
```

## 指标解释

- `item_understanding.pass@64`：64 个候选中是否包含 gold itemic pattern。
- `recommendation.overall_micro_pass@64`：每个 gold 推荐 token 独立计算命中后做 micro average。
- `recommendation.sample_any_pass@64`：样本中任意 gold token 命中即算通过。
- `user_related_items.f1`：模型筛出的相关历史 SID 与 gold SID 的集合 F1。
- `user_logic_chain.proxy_score`：规则 proxy，包含 JSON、events、SID、date、logic 等格式和 grounding 检查。
- `general_mcq.accuracy`：解析 `正确答案是 X`，与 `metadata.answer` 完全一致才得分。

## 注意事项

- 若要把本地 eval 当作可信验证集，训练时使用 `USE_EVAL_SPLIT=1` 排除验证样本。
- 常识问答可以用 `scripts/build_general_mcq_dataset.py` 从 CMMLU/MMLU 转换；`OneReason_General` 不是标准选择题数据。
- `懂用户` 的逻辑链官方可能使用 judge 或更复杂规则，本地第一版只做规则 proxy。
