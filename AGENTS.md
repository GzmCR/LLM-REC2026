# AGENTS.md

This repository is an LLM-Rec / OneReason competition workspace. It contains data
analysis scripts, augmented SFT dataset builders, and LLaMA-Factory SFT training
configuration for `OpenOneRec/OneReason-0.8B-pretrain-competition`.

For project-related explanations, prefer Chinese unless the user asks otherwise.

## Project Layout

- Official competition materials live under `Explorer_LLM_Rec_Competition/`.
  Do not use the old path with a leading space.
- Raw parquet data lives under `Explorer_LLM_Rec_Competition/data/`.
  The `OneReason_UserProfile ` subdirectory still has a trailing space; scripts
  that discover raw tables must tolerate that.
- Official SFT JSONL data lives under `dataset/`.
- Project docs live under `docs/`.
- Data script implementations live under `scripts/data/`.
  `scripts/analyze_data.py` and `scripts/build_augmented_datasets.py` are
  compatibility entrypoints and should stay usable.
- Training scripts live under `scripts/train/`.
- Training config lives under `configs/train/`.
- Conda environment specs live under `envs/`.

## Data Safety

Never commit or upload these local data/artifact directories:

- `dataset/`
- `Explorer_LLM_Rec_Competition/data/`
- `generated_dataset/`
- `outputs/`
- `train_data/`
- `train_output/`
- `third_party/`

Especially do not upload `OneReason_UserProfile` parquet shards. They are large
competition data files and must remain local.

Do not delete raw data, official SFT data, generated datasets, or analysis
outputs unless the user explicitly asks. It is fine to remove local cache files
such as `.DS_Store`, `__pycache__/`, and `.pyc`.

## Common Commands

Run data analysis:

```bash
python scripts/analyze_data.py \
  --sft-dir dataset \
  --raw-dir "Explorer_LLM_Rec_Competition/data" \
  --out-dir outputs/data_analysis
```

Build augmented SFT datasets:

```bash
python scripts/build_augmented_datasets.py \
  --raw-dir "Explorer_LLM_Rec_Competition/data" \
  --out-dir generated_dataset \
  --max-rows 50000 \
  --max-samples-per-task 5000 \
  --seed 2026
```

Prepare the server training environment and datasets:

```bash
bash scripts/train/run_all_train_setup.sh
```

Start full SFT on a GPU server:

```bash
bash scripts/train/20_train_full_sft.sh
```

## Training Notes

- Assume the local machine has no GPU; do not start real training locally.
- LLaMA-Factory is installed into `third_party/LLaMA-Factory/`.
- The default model is `OpenOneRec/OneReason-0.8B-pretrain-competition`.
- The default LLaMA-Factory dataset name is `onereason_sft_mixed`.
- `scripts/train/10_prepare_dataset.sh` uses official `dataset/` by default and
  automatically mixes in `generated_dataset/` when non-empty JSONL files exist.

## Change Guidelines

- Preserve existing command entrypoints unless the user explicitly asks for a
  breaking cleanup.
- When changing paths, update README, script defaults, `.gitignore`, and relevant
  docs together.
- For large parquet tables, use streaming, batching, or column-selective reads.
  Avoid loading the full `OneReason_UserProfile` table into memory.
- SFT dataset builders should output `{system, prompt, response}` JSONL. Convert
  to Alpaca format only in the training preparation step.
- Keep generated artifacts out of Git. Check `.gitignore` before adding new
  output directories.

## Validation

After changing Python or training scripts, run:

```bash
python -m py_compile scripts/data/analyze_data.py scripts/data/build_augmented_datasets.py scripts/analyze_data.py scripts/build_augmented_datasets.py scripts/train/11_register_dataset.py
bash -n scripts/train/*.sh
```

Check that the old leading-space official path has not returned:

```bash
rg -n -F '" Explorer_LLM_Rec_Competition' README.md docs scripts .gitignore configs envs
rg -n -F '\ Explorer_LLM_Rec_Competition' README.md docs scripts .gitignore configs envs
```

Before committing, check Git safety:

```bash
git status --short --ignored
```

Confirm the large data and generated artifact directories are still ignored.
