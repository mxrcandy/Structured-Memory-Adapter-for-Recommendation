"""Generate offline Engram-structure SPM and ITM prototype tables."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from .prototype_common import (
    batched_mean_project,
    hash_tokens,
    model_embeddings,
    next_prime,
    row_text,
    save_manifest,
    sid_spans,
    token_projection,
    iter_rows,
    load_pattern_cache,
    iter_pattern_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-size", type=int, default=65521)
    parser.add_argument("--spm-order", type=int, default=3)
    parser.add_argument("--itm-order", type=int, default=1, choices=(1, 2))
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--batch-patterns", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pattern-cache", default=None)
    args = parser.parse_args()
    if args.pattern_cache:
        manifest, _ = load_pattern_cache(args.pattern_cache)
        if manifest.get('spm_order', 0) < args.spm_order or manifest.get('itm_order', 0) < args.itm_order:
            raise ValueError('pattern cache orders are insufficient')

    size = next_prime(args.memory_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, embedding = model_embeddings(args.model, args.tokenizer)
    hidden = int(embedding.shape[1])
    projection = token_projection(hidden, args.head_dim, args.projection_seed)
    spm_orders = tuple(range(2, args.spm_order + 1))
    itm_orders = tuple(range(1, args.itm_order + 1))
    tables = {}
    counts = {}
    sums = {}
    for branch, orders in (("spm", spm_orders), ("itm", itm_orders)):
        sums[branch] = {order: torch.zeros(args.heads, size, args.head_dim)
                        for order in orders}
        counts[branch] = {order: torch.zeros(args.heads, size, dtype=torch.long)
                          for order in orders}

    pending = {("spm", order): [] for order in spm_orders}
    pending.update({("itm", order): [] for order in itm_orders})
    occurrences = Counter()
    distinct = Counter()
    samples = 0
    cached_spm = set()

    def flush(branch: str, order: int) -> None:
        key = (branch, order)
        if not pending[key]:
            return
        frequencies = Counter(pending[key]); pending[key].clear()
        items = iter(frequencies.items())
        while True:
            chunk = list(__import__('itertools').islice(items, args.batch_patterns))
            if not chunk: break
            patterns = [p for p, _ in chunk]
            distinct[key] += len(patterns)
            values = batched_mean_project(patterns, embedding, projection)
            weights = torch.tensor([w for _, w in chunk], dtype=torch.long)
            order_idx = order - 2 if branch == "spm" else order - 1
            for head in range(args.heads):
                buckets = torch.tensor([hash_tokens(p, order, head, size, order_idx) for p in patterns])
                sums[branch][order][head].index_add_(0, buckets, values * weights[:, None])
                counts[branch][order][head].index_add_(0, buckets, weights)

    if args.pattern_cache:
        cache_meta, cached = load_pattern_cache(args.pattern_cache)
        for branch, branch_orders in (("spm", spm_orders), ("itm", itm_orders)):
            for order in branch_orders:
                    key = (branch, order)
                    source = cached.get(f"{branch}{order}", Counter())
                    if branch == "itm":
                        # Cache stores complete abc items; Engram addresses and
                        # values use the same ab history as structure.py.
                        converted = Counter()
                        for pattern, count in source.items():
                            if len(pattern) != 3 * order:
                                raise ValueError(f"invalid complete abc ITM-{order} pattern: {pattern}")
                            ab = tuple(pattern[offset] for item_offset in range(order)
                                       for offset in (3 * item_offset, 3 * item_offset + 1))
                            converted[ab] += count
                        source = converted
                    pending[key] = source
                    occurrences[key] += sum(source.values())
                    flush(*key)
                    if branch == "spm":
                        cached_spm.add(order)
                    print(f"[engram-structure-cache] {branch}{order}", flush=True)
        samples = int(cache_meta.get("samples", 0))

    for file_index, file_count, _, row in ([] if args.pattern_cache else iter_rows(args.data_dir, args.max_samples)):
        spans = list(sid_spans(row_text(row), tokenizer))
        # SPM mirrors structure.py's phase windows: every complete n-gram
        # inside each SID span, including begin/end markers.
        for span in ([] if args.pattern_cache else spans):
            for position in range(1, len(span)):
                for order in spm_orders:
                    if position + 1 >= order:
                        pattern = tuple(span[position + 1 - order:position + 1])
                        pending[("spm", order)].append(pattern)
                        occurrences[("spm", order)] += 1
                        if len(pending[("spm", order)]) >= args.batch_patterns:
                            flush("spm", order)
            # ITM uses the completed item's (a,b) code pair, matching
            # structure.py histories[0]/histories[1].
        items = [tuple(span[-4:-1]) for span in spans if len(span) == 5]
        for index in range(1, len(items)):
            for order in itm_orders:
                if index >= order:
                    # Engram ITM address and value both use only ab codes:
                    # ITM-1=(a,b), ITM-2=(a1,b1,a2,b2). The cache retains c
                    # for Lngram-structure, so slice it away here.
                    pattern = tuple(token for item in items[index - order:index] for token in item[:2])
                    pending[("itm", order)].append(pattern)
                    occurrences[("itm", order)] += 1
                    if len(pending[("itm", order)]) >= args.batch_patterns:
                        flush("itm", order)
        samples += 1
        if samples % 1000 == 0:
            spm2_occ = occurrences[("spm", 2)]
            itm1_occ = occurrences[("itm", 1)]
            print(f"[engram-structure] files={file_index}/{file_count} samples={samples} "
                  f"spm2={spm2_occ} itm1={itm1_occ}", flush=True)

    for branch, orders in (("spm", spm_orders), ("itm", itm_orders)):
        for order in orders:
            flush(branch, order)
            table = sums[branch][order]
            count = counts[branch][order]
            mask = count > 0
            table[mask] /= count[mask].unsqueeze(-1)
            tables.setdefault(branch, {})[str(order)] = table

    manifest = {
        "format": "engram_structure_prototype_tokenemb_v1",
        "model": args.model,
        "tokenizer": args.tokenizer,
        "data_dir": args.data_dir,
        "source_filter": "RecIF_VideoRec_Pretrain",
        "spm_scope": "complete_sid_span_ngrams_2_to_order",
        "itm_scope": "previous_completed_item_ab_pair",
        "memory_size": size,
        "spm_order": args.spm_order,
        "itm_order": args.itm_order,
        "heads": args.heads,
        "hidden_size": hidden,
        "head_dim": args.head_dim,
        "projection": "column_normalized_random_token_embedding_mean",
        "projection_seed": args.projection_seed,
        "samples": samples,
        "occurrences": {f"{b}{o}": int(occurrences[(b, o)])
                        for b, orders in (("spm", spm_orders), ("itm", itm_orders)) for o in orders},
        "distinct_patterns": {f"{b}{o}": int(distinct[(b, o)])
                              for b, orders in (("spm", spm_orders), ("itm", itm_orders)) for o in orders},
        "collision_policy": "occurrence_weighted_mean",
    }
    name = (f"engram_structure_proto_tokenemb_randomproj{hidden}to{args.head_dim}_seed{args.projection_seed}"
            f"_spm2-{args.spm_order}_itm1-{args.itm_order}_heads{args.heads}_mem{size}.pt")
    torch.save({"manifest": manifest, "projection": projection, "tables": tables,
                "bucket_counts": counts}, output_dir / name)
    save_manifest(output_dir / (Path(name).stem + ".json"), manifest)
    print({"output": str(output_dir / name), **manifest}, flush=True)


if __name__ == "__main__":
    main()
