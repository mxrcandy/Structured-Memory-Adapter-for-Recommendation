"""Shared offline prototype-generation helpers.

The generators intentionally use only the RecIF_VideoRec_Pretrain rows and
only complete SID spans.  They are offline tools, not model components.
"""
from __future__ import annotations

import json
import re
import ast
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch

from onerec_llm.utils.data_utils import load_parquet_file


SID_TOKEN_RE = re.compile(r"<\|sid_begin\|>|<s_[abc]_\d+>|<\|sid_end\|>")


def next_prime(value: int) -> int:
    def is_prime(number: int) -> bool:
        if number < 2:
            return False
        if number % 2 == 0:
            return number == 2
        divisor = 3
        while divisor * divisor <= number:
            if number % divisor == 0:
                return False
            divisor += 2
        return True

    candidate = max(2, int(value))
    while not is_prime(candidate):
        candidate += 1
    return candidate


def hash_tokens(tokens: Iterable[int], order: int, head: int, size: int, order_idx: int | None = None) -> int:
    """Match Engram/structure.py's integer hash exactly."""
    # Runtime modules index ``order_salts`` by position in ``self.orders``;
    # pass that index explicitly for branches whose orders start at 2 or 1.
    # ``order_idx`` documents the runtime table index; the salt itself is
    # derived from the actual n-gram order (2/3 for SPM, 1/2 for ITM).
    salt_order = int(order)
    order_salt = 0x9E3779B1 + 0x85EBCA77 * salt_order
    head_salt = 0xC2B2AE3D + 0x27D4EB2F * int(head)
    # Use int64 tensor arithmetic so Python's arbitrary-precision integers do
    # not disagree with the overflow semantics of the runtime hash.
    hashed = torch.tensor(order_salt ^ head_salt, dtype=torch.long)
    for offset, token in enumerate(tokens):
        term = torch.tensor((int(token) + 1) * (head_salt + offset + 1), dtype=torch.long)
        hashed = torch.bitwise_xor(hashed * 0x9E3779B1, term)
    min_int = torch.iinfo(hashed.dtype).min
    safe_abs = torch.where(
        hashed == min_int,
        torch.full_like(hashed, min_int + 1),
        hashed.abs(),
    )
    return int(torch.remainder(safe_abs, int(size)).item())


def token_id_map(tokenizer):
    vocab = tokenizer.get_vocab()
    begin = vocab["<|sid_begin|>"]
    end = vocab["<|sid_end|>"]
    return vocab, begin, end


def row_text(row) -> str:
    segments = row.segments
    if isinstance(segments, str):
        segments = json.loads(segments)
    return "".join(item.get("text", "") for item in segments
                    if item.get("type") == "text")


def iter_rows(data_dir: str, max_samples: Optional[int] = None):
    samples = 0
    files = sorted(Path(data_dir).glob("*.parquet"))
    for file_index, path in enumerate(files, 1):
        parquet = load_parquet_file(str(path))
        for group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(group).to_pandas()
            for row in frame.itertuples(index=False):
                if row.source != "RecIF_VideoRec_Pretrain":
                    continue
                yield file_index, len(files), samples, row
                samples += 1
                if max_samples is not None and samples >= max_samples:
                    return


def sid_spans(text: str, tokenizer) -> Iterator[list[int]]:
    """Yield complete SID spans as token IDs, including begin/end markers."""
    vocab, begin, end = token_id_map(tokenizer)
    span = None
    for match in SID_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token == "<|sid_begin|>":
            span = [begin]
        elif token == "<|sid_end|>" and span is not None:
            span.append(end)
            yield span
            span = None
        elif span is not None:
            token_id = vocab.get(token)
            if token_id is not None:
                span.append(token_id)


def token_projection(hidden: int, output: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(hidden, output, generator=generator)
    return matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-8)


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def load_pattern_cache(cache_dir: str):
    """Load and merge sharded pattern counters produced by pattern_cache.py."""
    cache = Path(cache_dir)
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pattern cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    counters = {}
    for shard in sorted(cache.glob("patterns_part-*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        for name, values in payload.items():
            normalized = Counter()
            for key, count in values.items():
                if isinstance(key, str):
                    key = ast.literal_eval(key)
                if not isinstance(key, tuple):
                    key = tuple(key)
                normalized[key] += int(count)
            counters.setdefault(name, Counter()).update(normalized)
    if not counters:
        raise ValueError(f"Pattern cache contains no shards: {cache}")
    return manifest, counters


def iter_pattern_cache(cache_dir: str):
    """Yield ``(manifest, shard_name, payload)`` without merging shards."""
    cache = Path(cache_dir)
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pattern cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    shards = sorted(cache.glob("patterns_part-*.pt"))
    if not shards:
        raise ValueError(f"Pattern cache contains no shards: {cache}")
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        normalized = {}
        for name, values in payload.items():
            counter = Counter()
            for key, count in values.items():
                if isinstance(key, str):
                    key = ast.literal_eval(key)
                if not isinstance(key, tuple):
                    key = tuple(key)
                counter[key] += int(count)
            normalized[name] = counter
        yield manifest, shard.name, normalized


def model_embeddings(model_path: str, tokenizer_path: str, torch_dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch_dtype, device_map="cpu"
    )
    embedding = model.get_input_embeddings().weight.detach().cpu().float()
    return tokenizer, embedding


def batched_mean_project(patterns, embedding: torch.Tensor, projection: torch.Tensor):
    ids = torch.tensor(patterns, dtype=torch.long)
    values = embedding[ids].mean(dim=1).matmul(projection)
    return values
