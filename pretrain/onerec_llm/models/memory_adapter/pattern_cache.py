"""Build a reusable token-id pattern cache for all memory prototypes.

The cache is sharded by parquet file.  It stores only pattern tuples and
occurrence counts, so changing value dimensions or projections never requires
scanning the source dataset again.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .prototype_common import iter_rows, row_text, sid_spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spm-order", type=int, default=3)
    parser.add_argument("--itm-order", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counters = {}
    for branch, orders in (("spm", range(2, args.spm_order + 1)),
                           ("itm", range(1, args.itm_order + 1))):
        for order in orders:
            counters[(branch, order)] = Counter()
    current_file = None
    samples = 0

    def flush(file_index):
        nonlocal counters
        payload = {f"{branch}{order}": dict(counter)
                   for (branch, order), counter in counters.items() if counter}
        path = out / f"patterns_part-{file_index:05d}.pt"
        torch.save(payload, path)
        counters = {(branch, order): Counter() for branch, orders in
                    (("spm", range(2, args.spm_order + 1)),
                     ("itm", range(1, args.itm_order + 1))) for order in orders}

    for file_index, _, _, row in iter_rows(args.data_dir, args.max_samples):
        if current_file is not None and file_index != current_file:
            flush(current_file)
        current_file = file_index
        spans = list(sid_spans(row_text(row), tokenizer))
        for span in spans:
            for pos in range(1, len(span)):
                for order in range(2, args.spm_order + 1):
                    if pos + 1 >= order:
                        counters[("spm", order)][tuple(span[pos + 1 - order:pos + 1])] += 1
        # Store complete item codes for structure ITM prototypes:
        # ITM-1=(a,b,c), ITM-2=(a1,b1,c1,a2,b2,c2).
        summaries = [tuple(span[-4:-1]) for span in spans if len(span) >= 5]
        for pos in range(1, len(summaries)):
            for order in range(1, args.itm_order + 1):
                if pos >= order:
                    pattern = tuple(token for item in summaries[pos - order:pos] for token in item)
                    counters[("itm", order)][pattern] += 1
        samples += 1
        if samples % 1000 == 0:
            print(f"[pattern-cache] files={file_index} samples={samples}", flush=True)
    if current_file is not None:
        flush(current_file)
    manifest = {"format": "memory_pattern_cache_v1", "data_dir": args.data_dir,
                "tokenizer": args.tokenizer, "source_filter": "RecIF_VideoRec_Pretrain",
                "spm_order": args.spm_order, "itm_order": args.itm_order,
                "itm_pattern": "complete_abc_per_item",
                "samples": samples, "shard_count": len(list(out.glob("patterns_part-*.pt")))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
