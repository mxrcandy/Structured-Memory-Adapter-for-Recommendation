"""Generate deterministic latent SPM/ITM prototype tables for Lngram."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .lngram_prototype import batched_latent_addresses
from .prototype_common import (batched_mean_project, iter_rows, model_embeddings,
                               row_text, save_manifest, sid_spans,
                               token_projection, load_pattern_cache, iter_pattern_cache)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spm-order", type=int, default=3)
    parser.add_argument("--itm-order", type=int, default=1, choices=(1, 2))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--batch-patterns", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pattern-cache", default=None)
    parser.add_argument("--spm-prototype", default=None, help="Reuse matching Lngram SPM tables; scan only ITM from data")
    args = parser.parse_args()
    if args.pattern_cache:
        manifest = json.loads((Path(args.pattern_cache) / 'manifest.json').read_text())
        if manifest.get('spm_order', 0) < args.spm_order or manifest.get('itm_order', 0) < args.itm_order:
            raise ValueError('pattern cache orders are insufficient')
    tokenizer, embedding = model_embeddings(args.model, args.tokenizer)
    hidden = int(embedding.shape[1])
    routes = hidden // args.bits
    alphabet = 1 << args.bits
    projection = token_projection(hidden, args.memory_dim, args.projection_seed)
    spm_orders = tuple(range(2, args.spm_order + 1))
    itm_orders = tuple(range(1, args.itm_order + 1))
    orders = {"spm": spm_orders, "itm": itm_orders}
        # ITM uses one latent symbol per complete item.  The item symbol is
        # computed from the mean embedding of its cached (a,b) pair, which is the
    # offline proxy for the runtime hidden at c.  Thus ITM-1/2 retain the
    # compact R*16 and R*16^2 table sizes.
    address_order = lambda branch, order: order
    sums = {branch: {order: torch.zeros(routes * alphabet ** address_order(branch, order), args.memory_dim)
                     for order in branch_orders} for branch, branch_orders in orders.items()}
    counts = {branch: {order: torch.zeros(routes * alphabet ** address_order(branch, order), dtype=torch.long)
                       for order in branch_orders} for branch, branch_orders in orders.items()}
    pending = {(branch, order): [] for branch, branch_orders in orders.items()
               for order in branch_orders}
    occurrences = Counter()
    distinct = Counter()
    samples = 0
    if args.spm_prototype:
        payload = torch.load(args.spm_prototype, map_location='cpu', weights_only=False)
        m = payload['manifest']
        for name, expected in [('hidden_size', hidden), ('bits', args.bits), ('memory_dim', args.memory_dim), ('projection_seed', args.projection_seed)]:
            if m[name] != expected:
                raise ValueError(f'SPM prototype mismatch: {name}')
        if not torch.equal(projection, payload['value_projection']):
            raise ValueError('SPM projection mismatch')
        for order in spm_orders:
            sums['spm'][order] = payload['tables'][order].clone()
            counts['spm'][order] = payload['bucket_counts'][order].clone()
            occurrences[('spm', order)] = m['occurrences'][str(order)]
            distinct[('spm', order)] = m['distinct_patterns'][str(order)]
    ab_cache = {order: Counter() for order in itm_orders}

    def flush(branch, order):
        key = (branch, order)
        if not pending[key]:
            return
        frequencies = Counter(pending[key])
        pending[key].clear()
        distinct[key] += len(frequencies)
        items = iter(frequencies.items())
        while True:
          chunk = list(__import__('itertools').islice(items, args.batch_patterns))
          if not chunk:
            break
          batch = [pattern for pattern, _ in chunk]
          weights = torch.tensor([weight for _, weight in chunk], dtype=torch.long)
          if branch == 'itm':
              if any(len(p) != 3 * order for p in batch):
                  raise ValueError('ITM cache patterns must contain complete abc codes per item')
              proxies = embedding[torch.tensor(batch)].reshape(-1, order, 3, hidden).mean(2)
              proxy_ids = torch.arange(len(batch) * order).reshape(-1, order).tolist()
              addresses, vectors = batched_latent_addresses(proxy_ids, proxies.reshape(-1, hidden), args.bits, order)
          else:
              addresses, vectors = batched_latent_addresses(batch, embedding, args.bits, order)
          values = vectors.matmul(projection)
          indices = torch.arange(routes)[None, :] * alphabet ** order + addresses
          sums[branch][order].index_add_(0, indices.reshape(-1),
              (values[:, None, :].expand(-1, routes, -1) * weights[:, None, None]).reshape(-1, args.memory_dim))
          counts[branch][order].index_add_(0, indices.reshape(-1), weights[:, None].expand(-1, routes).reshape(-1))

    if args.pattern_cache and not args.spm_prototype:
        cache_meta, cached = load_pattern_cache(args.pattern_cache)
        for branch, branch_orders in orders.items():
            for order in branch_orders:
                key = (branch, order)
                pending[key] = cached.get(f"{branch}{order}", Counter())
                occurrences[key] += sum(pending[key].values())
                flush(*key)
                print(f"[lngram-structure-cache] {branch}{order}", flush=True)
        samples = int(cache_meta.get("samples", 0))
    if not args.pattern_cache or args.spm_prototype:
      for file_index, file_count, _, row in iter_rows(args.data_dir, args.max_samples):
        spans = list(sid_spans(row_text(row), tokenizer))
        for span in ([] if args.spm_prototype else spans):
            for position in range(1, len(span)):
                for order in spm_orders:
                    if position + 1 >= order:
                        pattern = tuple(span[position + 1 - order:position + 1])
                        pending[("spm", order)].append(pattern)
                        occurrences[("spm", order)] += 1
                        if len(pending[("spm", order)]) >= args.batch_patterns:
                            flush("spm", order)
        items = [tuple(span[-4:-2]) for span in spans if len(span) == 5]
        for index in range(1, len(items)):
            for order in itm_orders:
                if index >= order:
                    pattern = tuple(t for pair in items[index - order:index] for t in pair)
                    ab_cache[order][pattern] += 1
                    pending[("itm", order)].append(pattern)
                    occurrences[("itm", order)] += 1
                    if len(pending[("itm", order)]) >= args.batch_patterns:
                        flush("itm", order)
        samples += 1
        if samples % 1000 == 0:
            spm2_occ = occurrences[("spm", 2)]
            itm1_occ = occurrences[("itm", 1)]
            print(f"[lngram-structure] files={file_index}/{file_count} samples={samples} "
                  f"spm2={spm2_occ} itm1={itm1_occ}", flush=True)
    for branch, branch_orders in orders.items():
        for order in branch_orders:
            flush(branch, order)
            mask = counts[branch][order] > 0
            if not (args.spm_prototype and branch == 'spm'):
                sums[branch][order][mask] /= counts[branch][order][mask].unsqueeze(-1)
            if occurrences[(branch, order)] > 0 and not bool(mask.any()):
                raise RuntimeError(f"No latent addresses populated for {branch}{order} despite {occurrences[(branch, order)]} patterns")
            if not bool(mask.any()) or not torch.isfinite(sums[branch][order]).all():
                raise RuntimeError(f'Empty or nonfinite table: {branch}{order}')

    manifest = {
        "format": "lngram_structure_prototype_tokenemb_identityroute_v1",
        "model": args.model, "tokenizer": args.tokenizer, "data_dir": args.data_dir,
        "source_filter": "RecIF_VideoRec_Pretrain",
        "spm_scope": "complete_sid_span_ngrams_2_to_order",
        "itm_scope": "previous_completed_item_ab_proxy_latent_pattern",
        "spm_prototype": args.spm_prototype,
        "pattern_cache": args.pattern_cache,
        "itm_proxy": "mean(e(a),e(b),e(c)); one symbol per item",
        "hidden_size": hidden, "bits": args.bits, "alphabet": alphabet,
        "routes": routes, "spm_orders": list(spm_orders), "itm_orders": list(itm_orders),
        "memory_dim": args.memory_dim,
        "route_projection": "identity_first_hidden_coordinates",
        "value_projection": "column_normalized_random_token_embedding_mean",
        "projection_seed": args.projection_seed, "samples": samples,
        "occurrences": {f"{b}{o}": int(occurrences[(b, o)])
                        for b, branch_orders in orders.items() for o in branch_orders},
        "distinct_patterns": {f"{b}{o}": int(distinct[(b, o)])
                              for b, branch_orders in orders.items() for o in branch_orders},
        "collision_policy": "occurrence_weighted_mean_per_latent_address",
    }
    name = (f"lngram_structure_proto_tokenemb_identityroute_bits{args.bits}_seed{args.projection_seed}"
            f"_spm2-{args.spm_order}_itm1-{args.itm_order}_memdim{args.memory_dim}.pt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(ab_cache.values()):
        torch.save({'manifest': {'data_dir': args.data_dir, 'samples': samples, 'itm_scope': 'ab_pairs'},
                    'patterns': {f'itm{o}': dict(c) for o, c in ab_cache.items()}}, output_dir / 'itm_ab_patterns.pt')
    torch.save({"manifest": manifest, "value_projection": projection,
                "tables": sums, "bucket_counts": counts}, output_dir / name)
    save_manifest(output_dir / (Path(name).stem + ".json"), manifest)
    print({"output": str(output_dir / name), **manifest}, flush=True)


if __name__ == "__main__":
    main()
