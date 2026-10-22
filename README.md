# Structured Memory Adapter for Recommendation (SMAR)

This repository contains the SMAR implementation and the reproduction paths for both recommendation and general-capability evaluation. SMAR adds SID Prefix Memory (SPM) and Item Transition Memory (ITM) to a frozen Qwen3 backbone. See [the method note](docs/method.md) and the paper draft description for the method overview.

## What is included

- Training implementation and scripts under `pretrain/` (Qwen3 model integration, data loading, structured memory modules, checkpoint conversion, and focused tests).
- Amazon Beauty, Instruments, and Sports processed data under `data/`; each includes training shards and recommendation benchmark inputs/mappings. The three datasets occupy about 147 MB total.
- Recommendation benchmark implementation under `benchmarks/benchmark/`, using the tasks `amazon_beauty`, `amazon_instruments`, and `amazon_sports`.
- General benchmark wrapper under `benchmarks/lm_eval/`; the default `stage4_core` suite is MMLU-Pro, GSM8K, and IFEval. Official benchmark datasets are downloaded/cached separately.

Model weights, trained checkpoints, and general benchmark cache are not included. You need an expanded Qwen3-0.6B Itemic checkpoint matching the bundled data's four-level LETTER SID vocabulary. The recommendation test and training paths require CUDA/vLLM; lightweight structural tests can run on CPU.

## Setup

Use a Linux environment with Python 3.10, CUDA-enabled PyTorch, and the versions compatible with the vLLM release. For benchmark environments, see `benchmarks/README.md` and `benchmarks/requirements.txt`. Training also requires the distributed dependencies and MPI runtime used by your cluster; the launcher can be adapted to your site's scheduler.

Set paths in your shell rather than editing scripts:

```bash
export BASE_MODEL_DIR=/path/to/Qwen3-0.6B_itemic
export DATA_PROFILE=amazon_letter_beauty  # or amazon_letter_instruments / amazon_letter_sports
```

The repository data profiles and file lists are relocatable. No model checkpoint is downloaded automatically by the training scripts.

## Training

The published experiments start from an expanded Qwen3-0.6B Itemic checkpoint and train the structured adapter. A complete end-to-end run needs that external checkpoint and a multi-GPU environment. Set the paths and stage settings documented in `pretrain/scripts/e2e_common.sh`, then run:

```bash
DATA_PROFILE=amazon_letter_beauty \
STAGE1_CONVERTED_MODEL_DIR="$BASE_MODEL_DIR" \
RUN_REC_EVAL=false RUN_GENERAL_EVAL=false \
STAGE2_OUTPUT_DIR=/path/to/output/stage2 \
SFT_OUTPUT_DIR=/path/to/output/sft \
bash pretrain/scripts/train_e2e_adapter_structure.sh
```

Change `DATA_PROFILE` to run the same pipeline on Instruments or Sports. The pipeline uses the selected profile's Stage 2 and SFT Parquet shards and converts checkpoints to Hugging Face format. Output directories are set explicitly above so they stay outside Git. To submit Slurm evaluations after training, set `RUN_REC_EVAL=true` and `RUN_GENERAL_EVAL=true` and configure the corresponding scheduler/environment settings; by default this example disables automatic submission so the two evaluation paths below can be run manually. The Qwen3 backbone remains external; training checkpoints and logs should be written outside the repository.

## Recommendation evaluation

Provide a converted checkpoint containing the SMAR remote model code and run the matching test task. Example for Beauty:

```bash
MODEL_PATH=/path/to/converted \
DATA_PROFILE=amazon_letter_beauty \
TASK_TYPES=amazon_beauty \
NUM_GPUS=4 \
bash benchmarks/eval_amazon.sh
```

For Instruments use `DATA_PROFILE=amazon_letter_instruments TASK_TYPES=amazon_instruments`; for Sports use `DATA_PROFILE=amazon_letter_sports TASK_TYPES=amazon_sports`. You can set `SAMPLE_SIZE=20` for a short pipeline check. Full results use the benchmark's default protocol. The evaluator writes results next to the model run unless `OUTPUT_ROOT` and `LOG_ROOT` are set. The benchmark data is already included in each corresponding data profile.

## General-capability evaluation

Install `lm-eval==0.4.8` in the vLLM evaluation environment, prepare the official dataset cache once on a networked machine, and then run offline:

```bash
python benchmarks/lm_eval/prepare_data.py --data-dir data/general_benchmark

MODEL_PATH=/path/to/converted \
GENERAL_BENCHMARK_DATA_DIR="$PWD/data/general_benchmark" \
GENERAL_EVAL_SUITE=stage4_core \
DATA_PARALLEL_SIZE=4 \
bash benchmarks/eval_general.sh
```

`stage4_core` evaluates MMLU-Pro, GSM8K, and IFEval using the suite and protocol in `benchmarks/lm_eval/`. The cache is generated from the official Hugging Face datasets; it is intentionally excluded from Git. `SAMPLE_SIZE=20` limits evaluation for a quick check. `stage4_reasoning` and `stage4_full` are also provided.

## Checks

Run lightweight tests for model structure and the evaluation protocol (requires the test dependencies):

```bash
python -m pytest pretrain/tests/test_engram_structure.py pretrain/tests/test_exact_itm.py pretrain/tests/test_memory_adapter.py
python -m pytest benchmarks/tests/test_general_protocol.py benchmarks/tests/test_lm_eval_wrapper.py
```

The quality reproduction requires the external Qwen3 checkpoint, supported GPUs/vLLM, and full benchmark evaluation. The tests above validate core behavior and runner contracts; they do not claim to reproduce paper metrics.

## Data and attribution

The included recommendation data is processed from Amazon Review data for the Beauty, Musical Instruments, and Sports & Outdoors domains. The `data/*/README.md` and metadata files retain processing provenance. Respect the source dataset terms and cite the corresponding Amazon Reviews dataset. The code is derived from the OpenOneRec research codebase; benchmark code retains its included license and notices. A repository-wide license is omitted pending confirmation of all upstream code and data rights.
