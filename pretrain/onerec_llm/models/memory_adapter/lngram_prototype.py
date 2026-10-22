"""Generate a deterministic token-embedding prototype for Lngram tables.

The first offline version uses an identity route projector over the first
``routes * bits`` hidden coordinates.  This is deliberately recorded in the
manifest: training can later load the same route projector before using these
tables.  Each latent address receives the occurrence-weighted mean token
embedding projected to ADAPTER_HIDDEN_SIZE.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .prototype_common import (batched_mean_project, iter_rows, model_embeddings,
                               row_text, save_manifest, sid_spans,
                               token_projection, load_pattern_cache, iter_pattern_cache)


def latent_address(pattern, embedding, bits: int, order: int):
    alphabet = 1 << bits
    routes = embedding.shape[1] // bits
    token_values = embedding[torch.tensor(pattern, dtype=torch.long)]
    logits = token_values[:, :routes * bits].reshape(len(pattern), routes, bits)
    symbols = (logits >= 0).to(torch.long)
    addresses = torch.zeros(routes, dtype=torch.long)
    bit_values = 2 ** torch.arange(bits, dtype=torch.long)
    for offset in range(order):
        addresses += symbols[offset].mul(bit_values).sum(-1) * (alphabet ** offset)
    return addresses, token_values.mean(0)


def batched_latent_addresses(patterns, embedding, bits: int, order: int):
    """Vectorized identity-route addresses for a fixed-order pattern batch."""
    ids = torch.tensor(patterns, dtype=torch.long)
    token_values = embedding[ids].reshape(len(patterns), order, -1)
    routes = embedding.shape[1] // bits
    alphabet = 1 << bits
    symbols = (token_values[:, :, :routes * bits]
               .reshape(len(patterns), order, routes, bits) >= 0).to(torch.long)
    powers = (2 ** torch.arange(bits, dtype=torch.long)).view(1, 1, 1, bits)
    symbol_ids = (symbols * powers).sum(-1)
    order_powers = (alphabet ** torch.arange(order, dtype=torch.long)).view(1, order, 1)
    addresses = (symbol_ids * order_powers).sum(1)
    values = token_values.mean(1)
    return addresses, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ngram-order", type=int, default=3)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--batch-patterns", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pattern-cache", default=None)
    args = parser.parse_args()
    if args.pattern_cache:
        manifest = json.loads((Path(args.pattern_cache) / 'manifest.json').read_text())
        if manifest.get('spm_order', 0) < args.ngram_order: raise ValueError('pattern cache order is insufficient')
    tokenizer, embedding = model_embeddings(args.model, args.tokenizer)
    hidden = int(embedding.shape[1])
    if hidden % args.bits:
        raise ValueError("hidden size must be divisible by bits")
    routes = hidden // args.bits
    alphabet = 1 << args.bits
    projection = token_projection(hidden, args.memory_dim, args.projection_seed)
    orders = tuple(range(2, args.ngram_order + 1))
    sums = {order: torch.zeros(routes * alphabet ** order, args.memory_dim)
            for order in orders}
    counts = {order: torch.zeros(routes * alphabet ** order, dtype=torch.long)
              for order in orders}
    pending = {order: [] for order in orders}
    occurrences = Counter()
    distinct = Counter()
    samples = 0

    def flush(order):
        if not pending[order]:
            return
        frequencies = Counter(pending[order])
        pending[order].clear()
        items = iter(frequencies.items())
        while True:
          chunk = list(__import__('itertools').islice(items, args.batch_patterns))
          if not chunk: break
          patterns = [p for p, _ in chunk]
          distinct[order] += len(patterns)
          addresses, token_values = batched_latent_addresses(patterns, embedding, args.bits, order)
          values = token_values.matmul(projection)
          weights = torch.tensor([w for _, w in chunk], dtype=torch.long)
          route_ids = torch.arange(routes, dtype=torch.long).view(1, routes)
          indices = route_ids * (alphabet ** order) + addresses
          sums[order].index_add_(0, indices.reshape(-1),
                                 (values[:, None, :].expand(-1, routes, -1) * weights[:, None, None]).reshape(-1, args.memory_dim))
          counts[order].index_add_(0, indices.reshape(-1), weights[:, None].expand(-1, routes).reshape(-1))

    if args.pattern_cache:
        cache_meta = None
        for cache_meta, _, cached in iter_pattern_cache(args.pattern_cache):
            for order in orders:
                counter = cached.get(f"spm{order}", cached.get(str(order), {}))
                occurrences[order] += sum(counter.values())
                pending[order] = counter
                flush(order)
        samples = int(cache_meta.get("samples", 0)) if cache_meta else 0
    else:
      for file_index, file_count, _, row in iter_rows(args.data_dir, args.max_samples):
        for span in sid_spans(row_text(row), tokenizer):
            for position in range(1, len(span)):
                for order in orders:
                    if position + 1 >= order:
                        pattern = tuple(span[position + 1 - order:position + 1])
                        pending[order].append(pattern)
                        occurrences[order] += 1
                        if len(pending[order]) >= args.batch_patterns:
                            flush(order)
        samples += 1
        if samples % 1000 == 0:
            print(f"[lngram] files={file_index}/{file_count} samples={samples} "
                  f"occ2={occurrences[2]} occ3={occurrences[3]}", flush=True)
    for order in orders:
        flush(order)
        mask = counts[order] > 0
        sums[order][mask] /= counts[order][mask].unsqueeze(-1)
        if occurrences[order] > 0 and not bool(mask.any()):
            raise RuntimeError(f"No latent addresses populated for order {order} despite {occurrences[order]} patterns")
        if not bool(mask.any()) or not torch.isfinite(sums[order]).all():
            raise RuntimeError(f'Empty or nonfinite table: {order}')

    manifest = {
        "format": "lngram_prototype_tokenemb_identityroute_v1",
        "model": args.model, "tokenizer": args.tokenizer, "data_dir": args.data_dir,
        "source_filter": "RecIF_VideoRec_Pretrain",
        "scope": "complete_sid_span_ngrams_2_to_order",
        "hidden_size": hidden, "bits": args.bits, "alphabet": alphabet,
        "routes": routes, "orders": list(orders), "memory_dim": args.memory_dim,
        "route_projection": "identity_first_hidden_coordinates",
        "value_projection": "column_normalized_random_token_embedding_mean",
        "projection_seed": args.projection_seed, "samples": samples,
        "occurrences": {str(o): int(occurrences[o]) for o in orders},
        "distinct_patterns": {str(o): int(distinct[o]) for o in orders},
        "collision_policy": "occurrence_weighted_mean_per_latent_address",
    }
    name = (f"lngram_proto_tokenemb_identityroute_bits{args.bits}_seed{args.projection_seed}"
            f"_ng2-{args.ngram_order}_memdim{args.memory_dim}.pt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "value_projection": projection, "tables": sums,
                "bucket_counts": counts}, output_dir / name)
    save_manifest(output_dir / (Path(name).stem + ".json"), manifest)
    print({"output": str(output_dir / name), **manifest}, flush=True)


if __name__ == "__main__":
    main()
