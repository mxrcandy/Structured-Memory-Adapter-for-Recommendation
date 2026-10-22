# SMAR Reproducibility Repository Design

## Goal

Create a standalone, reviewable Git repository for the Structured Memory Adapter for Recommendation (SMAR) submission. A reader should be able to inspect the SPM/ITM implementation, understand the training and evaluation path, run the included checks, and reproduce the reported Amazon-domain experiments after providing the external backbone checkpoint.

## Scope

- Keep only the SMAR-related model implementation, Qwen3 integration, training/data-loading path, scripts/configuration, focused tests, and concise documentation.
- Include the three processed Amazon LETTER datasets already used in experiments: Beauty, Instruments, and Sports (approximately 147 MB total).
- Keep the training data manifests relocatable within the repository.
- Document external model/tokenizer preparation and exact experiment commands. Model weights and generated checkpoints remain external.
- Provide a CPU-friendly structural/unit test path and the actual GPU training/evaluation entrypoint; the smoke tests do not claim to reproduce model quality.

## Exclusions

- Do not copy pretrained or trained weights, general-domain corpora, raw multi-gigabyte Amazon archives, unrelated OpenOneRec modules, cached files, TensorBoard events, or run logs.
- Do not change the method implementation or reported experimental protocol while packaging it. Packaging-only path and documentation fixes are allowed and must be checked against the source.

## Proposed Layout

```text
README.md
LICENSE
requirements.txt
pretrain/onerec_llm/{models,memory-adapter support,data,training,losses,utils}/
pretrain/recipes/train_qwen3.py
pretrain/scripts/{model conversion,SMAR train/eval entrypoints}/
pretrain/tests/
process_data/output_amazon_letter_{beauty,instruments,sports}/
docs/method.md
docs/reproduction.md
```

Only modules imported by this path or necessary to configure and evaluate it should be included. Dataset content should be limited to training/SFT shards, manifests, SID/token configuration, benchmark inputs and item mappings needed by reproduction.

## Reproduction Flow

1. Install the documented Python/CUDA dependencies and provide the expanded Qwen3-0.6B checkpoint matching the four-level, 256-code-per-level LETTER SID tokenizer.
2. Select one of the three bundled dataset profiles.
3. Run the structure/adapter unit tests and a tiny synthetic model smoke test without loading external weights.
4. Launch the documented SMAR experiment entrypoint for training and benchmark evaluation with the external checkpoint.

## Validation

- Check all copied manifests resolve inside the new repository and all listed Parquet files exist.
- Run focused CPU-compatible SMAR, structure-routing, exact-ITM, and dataset-integrity tests.
- Check shell entrypoints and Python syntax.
- Verify Git ignores checkpoints, generated outputs, caches, and logs while tracking the three selected datasets.
- Compare copied model/data files with the workspace source and inspect the final repository status and size.

## Repository Setup and Attribution

The source workspace does not contain usable Git metadata, so the new repository is initialized independently. Preserve source-file notices and document attribution for included code and datasets. Add a repository-level license only when the applicable rights are verified; do not infer one from the reference repository.
