# lm-eval General Benchmark

This directory is the general-capability evaluation backend. It is deliberately
separate from the RecIF recommendation benchmark under `benchmarks/benchmark/`.

The default protocol uses explicit Qwen3 non-thinking,
0-shot, a shared 2048-token budget and full evaluation. Reruns replace the
selected run's old results under the original `benchmark_results/v1.0/` path.
The dataset version remains `v1.0`. `SAMPLE_SIZE` optionally limits samples.

The benchmark is intended to run offline. `prepare_data.py` must be run once on
a machine with network access, and the complete `raw_data/general_data/benchmark_data/hf_cache`
directory must be copied to the evaluation machine. `manifest.json` alone is not
the dataset. `eval_general.sh` sets `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
and `TRANSFORMERS_OFFLINE=1` by default, so missing cache files fail fast instead
of silently downloading data.

## Prepare official data

Use the `openonerec_lm_eval` environment and allow network access once:

```bash
conda activate openonerec_lm_eval
python benchmarks/lm_eval/prepare_data.py \
  --data-dir raw_data/general_data/benchmark_data
```

The script stores datasets in the Hugging Face cache below that directory and
writes `manifest.json`. It does not create the old RecIF parquet format.

## Run a suite

```bash
MODEL_PATH=/path/to/converted \
PYTHON_BIN=/path/to/openonerec_lm_eval/bin/python \
GENERAL_EVAL_SUITE=stage4_core \
bash benchmarks/eval_general.sh
```

Run the command from an allocated compute job. Scheduler submission files are site-specific and are not included in this repository.

Suites are YAML files in `suites/`. Add or remove official lm-eval task names
there without changing the runner.

## Sample data parallelism

`GENERAL_PARALLEL_MODE=data DATA_PARALLEL_SIZE=4` uses the recommendation
`RayMixin._initialize_ray_cluster` and `VllmWorker` directly. Each persistent
worker receives a disjoint interleaved shard of token requests. The driver
restores response order before lm-eval computes metrics. The shared protocol
bootstrap configures prompts, generation parameters and sample limits before
lm-eval constructs requests, identically for tensor and data execution modes.
Memory adapters use the recommendation worker's scheduler/history bridge.

`RAY_NUM_CPUS` defaults to `SLURM_CPUS_PER_TASK` when provided (or 8 otherwise). Each Ray actor retains its assigned
GPU visibility and isolated Transformers module cache. Worker readiness has
a 600-second timeout, adjustable with `GENERAL_WORKER_START_TIMEOUT`.

If Ray initialization stalls, the driver prints its Python stack every 120
seconds. Inspect the printed Ray temp directory's `session_latest/logs`,
especially `raylet.err`, `raylet.out`, and agent logs. On script exit logs are
copied to `<base_log_name>.ray_logs`. A local Ray start message alone does not
prove driver registration or model initialization succeeded.
