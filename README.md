# Structured Memory Adapter for Recommendation (SMAR)

Our goal is to inject recommendation capability into a general-purpose language model while preserving its general capabilities. Structured Memory Adapter for Recommendation (SMAR) pursues this goal by adding two structured memory branches to a frozen Qwen3 backbone: SID Prefix Memory (SPM) for item-internal semantic-code patterns, and Item Transition Memory (ITM) for transitions from the most recently completed historical item. This repository contains the training implementation, processed Amazon recommendation data, and runners for recommendation and general-capability evaluation.

## Method Overview

Generative recommenders predict an item's semantic identifier (SID) autoregressively. SMAR augments the frozen backbone with two conditional memories:

- **SID Prefix Memory (SPM)** retrieves learned information for valid prefixes within the SID currently being generated. It provides item-internal code-combination information when predicting later SID codes.
- **Item Transition Memory (ITM)** retrieves transition information using eligible prefixes from the most recently completed historical item. The history state updates at item boundaries, so the memory can inform prediction of the next item's first code.

Each memory branch projects its retrieved representation to the backbone hidden size and uses a context gate to control its residual contribution. Ordinary text and invalid SID positions do not activate the corresponding memory. The branches are trained with the autoregressive recommendation objective while the Qwen3 backbone remains frozen.

## Example Data

Three processed Amazon Reviews domains are included under `process_data/`:

| Domain | Stage 1 catalog items | Stage 2 histories | SFT examples | Recommendation test examples |
| --- | ---: | ---: | ---: | ---: |
| Beauty | 12,101 | 22,363 | 22,363 | 22,363 |
| Musical Instruments | 9,922 | 24,772 | 24,772 | 24,772 |
| Sports & Outdoors | 18,357 | 35,598 | 35,598 | 35,598 |

Each Stage 1 row contains an item's four-level LETTER SID and text. Stage 2 uses each user's interaction history except the final event. SFT predicts the penultimate event from the earlier history, and recommendation evaluation predicts the final event from the preceding history. Dataset-specific processing details and manifests are documented in each `process_data/output_amazon_letter_*/README.md` and `summary.json`.

The data preparation follows the Amazon recommendation processing flow in [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec), adapted for the included four-level LETTER SIDs and SMAR benchmark tasks. The bundled datasets are processed artifacts; source Amazon Reviews data and model checkpoints are not included.

## Installation

The repository targets Linux and Python 3.10. Training, recommendation evaluation, and general-capability evaluation use separate Conda environments because their PyTorch, Transformers, and vLLM requirements differ. The commands below follow the versions used in the training and evaluation plans. Adjust the PyTorch wheel for the CUDA runtime supported by your cluster if it is not CUDA 12.1.

### Training and data environment

~~~bash
conda create -n smar_train python=3.10 -y
conda activate smar_train

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.53 safetensors pyarrow pandas numpy tqdm huggingface_hub easydict torchao==0.10 sortedcontainers
pip install accelerate torchdata tensorboard webdataset rich fastparquet flash-attn einops
~~~

`flash-attn` may need to be built for the cluster's CUDA toolkit and compiler. Confirm that the environment sees the allocated GPUs before training:

~~~bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
~~~

Training also requires an MPI runtime with `mpirun` available in your shell. The training scripts use it to launch one worker per GPU.

### Recommendation benchmark environment

This environment provides the OpenOneRec-derived recommendation benchmark and vLLM 0.7.3:

~~~bash
conda create -n smar_benchmark python=3.10 -y
conda activate smar_benchmark
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.7.3 transformers==4.52.0
pip install -r benchmarks/requirements.txt
~~~

Run `benchmarks/eval_amazon.sh` from an allocated GPU job; the runner adds the local benchmark source to `PYTHONPATH`, so a separate package installation is not needed. The three recommendation datasets and their benchmark inputs are included under `process_data/`.

### General-capability benchmark environment

Keep lm-eval separate from the recommendation benchmark environment:

~~~bash
conda create -n smar_lm_eval python=3.10 -y
conda activate smar_lm_eval
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.52.0 vllm==0.7.3 datasets==3.6.0 \
  accelerate==1.8.1 sentencepiece protobuf pyarrow math-verify==0.8.0
pip install "lm-eval[vllm]==0.4.8" "lm-eval[ifeval]==0.4.8" "lm-eval[math]==0.4.8"
~~~

Prepare the official general benchmark datasets on a machine with network access, then copy the complete cache directory to the evaluation machine. The official datasets are not bundled:

~~~bash
python benchmarks/lm_eval/prepare_data.py --data-dir data/general_benchmark
~~~

See [the general benchmark setup notes](benchmarks/lm_eval/README.md) for cache and offline evaluation details. Run `benchmarks/eval_general.sh` from an allocated GPU job after setting `GENERAL_BENCHMARK_DATA_DIR` to the prepared cache location.

### Model Preparation

Download the Qwen3-0.6B checkpoint separately; model weights are not bundled with this repository. In the training environment, expand its vocabulary to support the four-level LETTER SIDs used by the included datasets, with 256 codes per level:

```bash
conda activate smar_train

HF_MODEL_DIR=/path/to/Qwen3-0.6B \
OUTPUT_MODEL_DIR=/path/to/Qwen3-0.6B_itemic \
ITEMIC_LAYER_N=4 \
VOCAB_SIZE_PER_LAYER=256 \
bash pretrain/scripts/expand_qwen3_vocab.sh
```

Set `HF_MODEL_DIR` to the downloaded checkpoint and `OUTPUT_MODEL_DIR` to the destination for the expanded checkpoint. Use the expanded checkpoint as `MODEL_DIR` for Stage 1 training below.

## Train

First train Stage 1 from the expanded Qwen3 checkpoint. Set `OUTPUT_DIR` outside the repository:

```bash
DATA_PROFILE=amazon_letter_beauty \
MODEL_DIR=/path/to/Qwen3-0.6B_itemic \
OUTPUT_DIR=/path/to/output/stage1 \
bash pretrain/scripts/train_stage1_itemic.sh
```

Convert its latest checkpoint to Hugging Face format (replace `STEP` with the number in `OUTPUT_DIR/latest`):

```bash
bash pretrain/scripts/convert_checkpoint_to_hf.sh \
  /path/to/Qwen3-0.6B_itemic \
  /path/to/output/stage1 \
  STEP
```

The end-to-end pipeline then runs Stage 2 training, SFT, and checkpoint conversion. Set `STAGE1_CONVERTED_MODEL_DIR` to the converted Stage 1 checkpoint under `stage1/step<STEP>/global_step<STEP>/converted`, and use output paths outside the repository:

```bash
DATA_PROFILE=amazon_letter_beauty \
STAGE1_CONVERTED_MODEL_DIR=/path/to/output/stage1/step<STEP>/global_step<STEP>/converted \
STAGE2_OUTPUT_DIR=/path/to/output/stage2 \
SFT_OUTPUT_DIR=/path/to/output/sft \
bash pretrain/scripts/train_e2e_adapter_structure.sh
```

Use `amazon_letter_instruments` or `amazon_letter_sports` in `DATA_PROFILE` to train on another included domain. The profiles select the corresponding Stage 1, Stage 2, and SFT data shards. The pipeline ends after checkpoint conversion; evaluation is run separately using the commands below.

## Evaluate Recommendation

Point `MODEL_PATH` to the converted SMAR checkpoint. This Beauty example runs the `amazon_beauty` recommendation task:

```bash
MODEL_PATH=/path/to/converted \
DATA_PROFILE=amazon_letter_beauty \
TASK_TYPES=amazon_beauty \
NUM_GPUS=4 \
bash benchmarks/eval_amazon.sh
```

For Instruments, use `DATA_PROFILE=amazon_letter_instruments TASK_TYPES=amazon_instruments`; for Sports, use `DATA_PROFILE=amazon_letter_sports TASK_TYPES=amazon_sports`. The processed benchmark data is included in each domain directory. Set `SAMPLE_SIZE=20` for a short run; omit it for the full benchmark protocol.

## Evaluate General Capabilities

Prepare the official benchmark dataset cache on a networked machine, then run evaluation from the allocated compute job:

```bash
python benchmarks/lm_eval/prepare_data.py --data-dir data/general_benchmark

MODEL_PATH=/path/to/converted \
GENERAL_BENCHMARK_DATA_DIR="$PWD/data/general_benchmark" \
GENERAL_EVAL_SUITE=stage4_core \
DATA_PARALLEL_SIZE=4 \
bash benchmarks/eval_general.sh
```

The `stage4_core` suite contains MMLU-Pro, GSM8K, and IFEval. The official datasets and cache are not bundled. `stage4_reasoning` and `stage4_full` are also available; see `benchmarks/lm_eval/` for suite definitions and protocol details.

## Repository Layout

~~~text
.
├── pretrain/                              # training code, model integration, and launch scripts
│   ├── onerec_llm/
│   │   ├── data/                          # Parquet datasets, readers, and dataloaders
│   │   ├── models/
│   │   │   ├── memory_adapter/            # SPM/ITM implementation
│   │   │   └── qwen3/                     # Qwen3 model and config integration
│   │   ├── training/                      # distributed training and checkpoint utilities
│   │   └── utils/                         # data, worker, and distributed helpers
│   ├── recipes/train_qwen3.py             # training entry point
│   ├── examples/dataset_config/           # training dataset config templates
│   ├── tools/model_converter/            # vocabulary expansion and HF checkpoint conversion
│   └── scripts/                           # vocab expansion, data profiles, train and convert scripts
├── process_data/                          # three processed Amazon domains and data profiles
│   └── output_amazon_letter_{beauty,instruments,sports}/
│       ├── tasks/                          # Stage 1, Stage 2, and SFT source parquet files
│       ├── split_data_pretrain_item_understand/ # sharded Stage 1 data + file_list.json
│       ├── split_data_pretrain/            # sharded Stage 2 data + file_list.json
│       ├── split_data_sft/                 # sharded SFT data + file_list.json
│       ├── intermediate/                   # item catalog and interaction sequences
│       ├── benchmark_data/                 # test parquet and SID-to-item mappings
│       ├── special_tokens.json             # SID and task token definitions
│       ├── README.md                       # domain data and processing notes
│       └── summary.json                    # counts and processing metadata
├── benchmarks/
│   ├── benchmark/                          # recommendation runner and task evaluators
│   ├── scripts/ray-vllm/                   # distributed vLLM generation support
│   ├── lm_eval/
│   │   ├── suites/                         # stage4_core/reasoning/full task lists
│   │   ├── prepare_data.py                 # cache official general benchmark datasets
│   │   └── wrapper.py                      # lm-eval protocol and execution wrapper
│   ├── eval_amazon.sh                      # recommendation evaluation entry point
│   ├── eval_general.sh                     # general-capability evaluation entry point
│   ├── requirements.txt                    # pinned recommendation benchmark dependencies
│   └── LICENSE                             # upstream benchmark license notice
└── README.md                               # method overview and reproduction guide
~~~

Each Amazon domain directory has the same organization. `split_data_*` contains sharded Parquet files and a `file_list.json` manifest; see the domain README for dataset-specific counts, schemas, and processing details.

## Acknowledgements

SMAR builds on recommendation training and evaluation code from [Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec). We adapt its Amazon data processing flow to the LETTER SID datasets and SMAR tasks in this repository. Please acknowledge OpenOneRec and cite the source Amazon Reviews datasets when using this code or data.
