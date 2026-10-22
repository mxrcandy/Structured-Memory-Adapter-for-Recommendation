"""Build Engram prototype tables from Stage-1 token embeddings."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from onerec_llm.utils.data_utils import load_parquet_file
from .prototype_common import load_pattern_cache, iter_pattern_cache
SID_TOKEN_RE = re.compile(r"<\|sid_begin\|>|<s_[abc]_\d+>|<\|sid_end\|>")

def next_prime(v):
    def prime(n):
        if n < 2: return False
        if n % 2 == 0: return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0: return False
            d += 2
        return True
    n = max(2, int(v))
    while not prime(n): n += 1
    return n

def hash_ngram(tokens, order, head, size):
    salt = (0x9E3779B1 + 0x85EBCA77 * order) ^ (0xC2B2AE3D + 0x27D4EB2F * head)
    hashed = salt
    for offset, token in enumerate(tokens):
        hashed = ((hashed * 0x9E3779B1) ^ ((int(token) + 1) * (0xC2B2AE3D + 0x27D4EB2F * head + offset + 1)))
    return abs(hashed) % size

def sid_patterns(ids, begin, end, max_order):
    start = None
    for i, token in enumerate(ids):
        if token == begin: start = i
        elif token == end and start is not None:
            span = ids[start:i + 1]
            for pos in range(1, len(span)):
                for order in range(2, max_order + 1):
                    if pos + 1 >= order:
                        yield order, tuple(span[pos + 1 - order:pos + 1])
            start = None

def sid_patterns_from_text(text, tokenizer, max_order):
    begin = tokenizer.convert_tokens_to_ids('<|sid_begin|>')
    end = tokenizer.convert_tokens_to_ids('<|sid_end|>')
    vocab = tokenizer.get_vocab()
    span = None
    for match in SID_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token == '<|sid_begin|>':
            span = [begin]
        elif token == '<|sid_end|>' and span is not None:
            span.append(end)
            yield from sid_patterns(span, begin, end, max_order)
            span = None
        elif span is not None:
            token_id = vocab.get(token)
            if token_id is not None: span.append(token_id)

def projection(h, d, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(h, d, generator=g)
    return x / x.norm(dim=0, keepdim=True).clamp_min(1e-8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--tokenizer', default=None); p.add_argument('--data-dir', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--memory-size', type=int, default=65521); p.add_argument('--ngram-order', type=int, default=3)
    p.add_argument('--hash-heads', type=int, default=8); p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--projection-seed', type=int, default=42); p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--pattern-cache', default=None)
    p.add_argument('--batch-patterns', type=int, default=8192)
    a = p.parse_args(); size = next_prime(a.memory_size); outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    if a.pattern_cache:
        manifest, _ = load_pattern_cache(a.pattern_cache)
        if manifest.get('spm_order', 0) < a.ngram_order: raise ValueError('pattern cache order is insufficient')
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, trust_remote_code=True)
    begin, end = tok.convert_tokens_to_ids('<|sid_begin|>'), tok.convert_tokens_to_ids('<|sid_end|>')
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.float32, device_map='cpu')
    emb = model.get_input_embeddings().weight.detach().cpu().float(); h = emb.shape[1]; proj = projection(h, a.head_dim, a.projection_seed)
    sums = {n: torch.zeros(a.hash_heads, size, a.head_dim, dtype=torch.float64) for n in range(2, a.ngram_order + 1)}
    counts = {n: torch.zeros(a.hash_heads, size, dtype=torch.long) for n in range(2, a.ngram_order + 1)}
    pattern_sets = {n: set() for n in range(2, a.ngram_order + 1)}; occurrences = {n: 0 for n in range(2, a.ngram_order + 1)}
    pending = {n: [] for n in range(2, a.ngram_order + 1)}; samples = 0
    def flush(order):
        if not pending[order]: return
        freq = Counter(pending[order]); pending[order].clear()
        items = iter(freq.items())
        while True:
            chunk = list(__import__('itertools').islice(items, a.batch_patterns))
            if not chunk: break
            patterns = [x for x, _ in chunk]
            weights = torch.tensor([w for _, w in chunk], dtype=torch.long)
            pattern_sets[order].update(patterns)
            ids = torch.tensor(patterns, dtype=torch.long)
            values = emb[ids].mean(1).matmul(proj).double()
            for head in range(a.hash_heads):
                buckets = torch.tensor([hash_ngram(x, order, head, size) for x in patterns], dtype=torch.long)
                sums[order][head].index_add_(0, buckets, values * weights[:, None])
                counts[order][head].index_add_(0, buckets, weights)
    if a.pattern_cache:
        cache_meta = None
        for cache_meta, _, cached in iter_pattern_cache(a.pattern_cache):
            for order in pending:
                counter = cached.get(f"spm{order}", cached.get(str(order), {}))
                occurrences[order] += sum(counter.values())
                pending[order] = counter
                flush(order)
        samples = int(cache_meta.get('samples', 0)) if cache_meta else 0
    files = [] if a.pattern_cache else sorted(Path(a.data_dir).glob('*.parquet'))
    for file_index, path in enumerate(files, 1):
        pq = load_parquet_file(str(path))
        for group in range(pq.num_row_groups):
            frame = pq.read_row_group(group).to_pandas()
            for row in frame.itertuples(index=False):
                if row.source != 'RecIF_VideoRec_Pretrain': continue
                segs = json.loads(row.segments); text = ''.join(x.get('text','') for x in segs if x.get('type') == 'text')
                for order, pattern in sid_patterns_from_text(text, tok, a.ngram_order):
                    pending[order].append(pattern); occurrences[order] += 1
                    if len(pending[order]) >= a.batch_patterns: flush(order)
                samples += 1
                if a.max_samples and samples >= a.max_samples: break
            if a.max_samples and samples >= a.max_samples: break
        print(f'[prototype] files={file_index}/{len(files)} samples={samples} occ2={occurrences[2]} occ3={occurrences[3]}', flush=True)
        if a.max_samples and samples >= a.max_samples: break
    for order in pending: flush(order)
    tables, bcounts, bpatterns = {}, {}, {}
    for order in sums:
        c = counts[order]; table = torch.zeros_like(sums[order]); mask = c > 0
        table[mask] = sums[order][mask] / c[mask].unsqueeze(-1)
        tables[str(order)] = table.float(); bcounts[str(order)] = c
        bpatterns[str(order)] = torch.zeros_like(c, dtype=torch.int32)
    manifest = {'format':'engram_prototype_tokenemb_v1','model':str(a.model),'tokenizer':str(a.tokenizer or a.model),'data_dir':str(a.data_dir),'source_filter':'RecIF_VideoRec_Pretrain','scope':'sid_span_all_complete_ngrams','include_sid_begin':True,'include_sid_end':True,'include_text':False,'cross_span':False,'memory_size':size,'ngram_order':a.ngram_order,'hash_heads':a.hash_heads,'hidden_size':h,'head_dim':a.head_dim,'projection':'column_normalized_random','projection_seed':a.projection_seed,'samples':samples,'pattern_counts':{str(k):len(v) for k,v in pattern_sets.items()},'pattern_occurrences':{str(k):int(v) for k,v in occurrences.items()},'collision_policy':'occurrence_weighted_mean','batch_patterns':a.batch_patterns}
    name = f'engram_proto_tokenemb_randomproj{h}to{a.head_dim}_seed{a.projection_seed}_ng2-{a.ngram_order}_heads{a.hash_heads}_mem{size}.pt'
    torch.save({'manifest':manifest,'projection':proj,'tables':tables,'bucket_counts':bcounts,'bucket_pattern_counts':bpatterns}, outdir / name)
    (outdir / (Path(name).stem + '.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False)); print(json.dumps({'output':str(outdir/name), **manifest}, ensure_ascii=False))
if __name__ == '__main__': main()
