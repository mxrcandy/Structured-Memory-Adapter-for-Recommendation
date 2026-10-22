"""Download official lm-eval datasets into the project's HF cache."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATASETS = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
    # lm-eval 0.4.8 resolves this task as the unqualified ``gsm8k`` dataset.
    # Cache that exact identifier so HF offline mode can find it.
    "gsm8k": ("gsm8k", "main", "test"),
    "ifeval": ("google/IFEval", None, "train"),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "minerva_math500": ("HuggingFaceH4/MATH-500", None, "test"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="raw_data/general_data/benchmark_data")
    parser.add_argument("--tasks", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(data_dir / "hf_cache"))
    from datasets import load_dataset

    manifest = {"prepared_at": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for task in args.tasks:
        if task not in DATASETS:
            raise ValueError(f"Unsupported data preparation task: {task}")
        path, config, split = DATASETS[task]
        dataset = load_dataset(path, config, split=split)
        manifest["datasets"][task] = {
            "path": path,
            "config": config,
            "split": split,
            "num_rows": len(dataset),
        }
        print(f"{task}: {len(dataset)} rows")
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
