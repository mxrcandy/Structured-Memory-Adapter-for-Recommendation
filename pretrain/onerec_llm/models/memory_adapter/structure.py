"""Causal SID memory with optional corpus-selected exact ITM source slots."""
from __future__ import annotations

import math
import hashlib
import json
import os
import re
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .base import BaseMemoryAdapter


def _structure_hidden_size(config, branch, hidden_size):
    branch_size = getattr(config, f'adapter_{branch}_hidden_size', None)
    return branch_size or getattr(config, 'adapter_hidden_size', None) or hidden_size


def configure_structure_tokens(config, tokenizer):
    config.adapter_structure_version = 1
    vocab = tokenizer.get_vocab()
    config.adapter_sid_begin_id = vocab['<|sid_begin|>']
    config.adapter_sid_end_id = vocab['<|sid_end|>']
    ranges = []
    levels = sorted({m.group(1) for token in vocab
                     if (m := re.fullmatch(r'<s_([a-z])_\d+>', token))})
    if len(levels) < 2 or levels != list('abcdefghijklmnopqrstuvwxyz'[:len(levels)]):
        raise ValueError('Structured memory requires consecutive SID levels starting at a')
    for level in levels:
        ids = sorted(v for k, v in vocab.items() if re.fullmatch(r'<s_' + level + r'_\d+>', k))
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError('Structured memory requires a contiguous token range for SID ' + level)
        ranges.append([ids[0], ids[-1]])
    config.adapter_sid_code_ranges = ranges


def restore_exact_itm_for_resume(config, checkpoint_path):
    """Restore immutable slot identities before constructing a resumed model."""
    path = Path(checkpoint_path).resolve()
    saved = None
    for parent in [path] + list(path.parents)[:3]:
        candidate = parent / 'structure_adapter_config.json'
        if candidate.is_file():
            saved = json.loads(candidate.read_text())
            break
    mode = getattr(config, 'adapter_itm_mode', 'default')
    if saved is None:
        if mode in ('exact', 'successor_prior', 'successor_prior_learned'):
            raise ValueError('Exact resume requires checkpoint structure_adapter_config.json')
        return

    def effective_branch_dim(values, branch, fallback_hidden):
        return (values.get(f'adapter_{branch}_hidden_size') or values.get('adapter_hidden_size')
                or values.get('hidden_size') or fallback_hidden)

    current_config = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    saved_type = saved.get('adapter_type')
    if saved_type is None and (saved.get('adapter_structure_version') is not None or
                               saved.get('adapter_sid_code_ranges') is not None or
                               saved.get('adapter_itm_mode') in
                               ('exact', 'successor_prior', 'successor_prior_learned')):
        saved_type = 'engram_structure'
    if config.adapter_type == 'engram_structure' and saved_type == 'engram_structure':
        for branch, enabled_key in (('spm', 'adapter_use_spm'), ('itm', 'adapter_use_itm')):
            if not (saved.get(enabled_key, True) and current_config.get(enabled_key, True)):
                continue
            saved_has_dimension = any(
                saved.get(key) is not None
                for key in (f'adapter_{branch}_hidden_size', 'adapter_hidden_size', 'hidden_size'))
            if not saved_has_dimension:
                continue
            if effective_branch_dim(saved, branch, config.hidden_size) != effective_branch_dim(
                    current_config, branch, config.hidden_size):
                raise ValueError(f'Cannot change adapter_{branch}_hidden_size while resuming')

    try:
        aux_weight = float(saved.get('adapter_itm_aux_loss_weight', 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid saved adapter_itm_aux_loss_weight') from exc
    aux_head = saved.get('adapter_itm_aux_head', aux_weight > 0)
    if (not math.isfinite(aux_weight) or aux_weight < 0 or type(aux_head) is not bool or
            (aux_weight > 0 and not aux_head)):
        raise ValueError('Invalid saved ITM auxiliary weight/head configuration')
    if aux_head and (mode != 'exact' or config.adapter_type != 'engram_structure' or
                     not getattr(config, 'adapter_use_itm', True)):
        raise ValueError('Cannot drop the saved exact ITM auxiliary head while resuming')
    if aux_head:
        # DCP permits extra checkpoint tensors, so check head identities before
        # loading: dropping one selected layer would otherwise silently lose it.
        def selected_layers(value):
            if value is None or (isinstance(value, str) and value.strip() in ('', 'all')):
                return set(range(config.num_hidden_layers))
            if isinstance(value, str):
                value = [part.strip() for part in value.split(',') if part.strip()]
            return {int(layer) for layer in value if 0 <= int(layer) < config.num_hidden_layers}

        if 'adapter_layers' in saved and selected_layers(saved['adapter_layers']) != selected_layers(
                getattr(config, 'adapter_layers', None)):
            raise ValueError('Cannot change adapter_layers for saved ITM auxiliary heads while resuming')
        for key in ('adapter_type', 'adapter_use_itm', 'adapter_itm_mode', 'adapter_itm_order',
                    'adapter_topk', 'adapter_sid_code_ranges', 'adapter_sid_begin_id', 'adapter_sid_end_id'):
            if key in saved and saved[key] != getattr(config, key, None):
                raise ValueError(f'Cannot change {key} for saved ITM auxiliary heads while resuming')
    config.adapter_itm_aux_loss_weight = aux_weight
    config.adapter_itm_aux_head = aux_head
    if saved.get('adapter_itm_mode', 'default') != mode:
        raise ValueError('Cannot change ITM mode while resuming a checkpoint')
    if mode in ('exact', 'successor_prior', 'successor_prior_learned'):
        mapping = saved.get('adapter_itm_mapping')
        if not mapping:
            raise ValueError('Exact resume checkpoint is missing slot identities')
        existing = getattr(config, 'adapter_itm_mapping', None)
        if existing is not None and existing != mapping:
            raise ValueError('Resume checkpoint and model directory use different exact slots')
        config.adapter_itm_mapping = mapping
        config.adapter_itm_pattern_manifest = saved.get('adapter_itm_pattern_manifest')
        if mode in ('successor_prior', 'successor_prior_learned'):
            counts = saved.get('adapter_itm_prior_counts')
            if not counts:
                raise ValueError('Successor prior resume is missing saved counts')
            existing_counts = getattr(config, 'adapter_itm_prior_counts', None)
            if existing_counts is not None and existing_counts != counts:
                raise ValueError('Cannot change successor prior counts while resuming')
            for key in ('adapter_layers', 'adapter_use_itm', 'adapter_topk',
                        'adapter_use_alpha_layer', 'adapter_fixed_context_gate',
                        'adapter_sid_code_ranges', 'adapter_sid_begin_id', 'adapter_sid_end_id'):
                if key in saved and saved[key] != getattr(config, key, None):
                    raise ValueError(f'Cannot change {key} while resuming successor prior')
            config.adapter_itm_prior_counts = counts


def validate_successor_prior(config):
    """Validate the compact, checkpoint-owned distribution without allocating tensors."""
    if (config.adapter_type != 'engram_structure' or config.adapter_itm_order != 1
            or getattr(config, 'adapter_itm_aux_loss_weight', 0) > 0
            or getattr(config, 'adapter_itm_aux_head', False)):
        raise ValueError('successor_prior requires engram_structure/order1 and no auxiliary head')
    bases = [hi-lo+1 for lo, hi in config.adapter_sid_code_ranges[:2]]
    mapping = getattr(config, 'adapter_itm_mapping', None)
    counts = getattr(config, 'adapter_itm_prior_counts', None)
    if not isinstance(mapping, list) or len(mapping) != bases[0]*bases[1]:
        raise ValueError('Invalid successor prior mapping')
    if any(type(x) is not int or x < -1 for x in mapping):
        raise ValueError('Invalid successor prior slot')
    slots = sorted(x for x in mapping if x >= 0)
    if not slots or slots != list(range(len(slots))) or not isinstance(counts, list) or len(counts) != len(slots):
        raise ValueError('Successor prior mapping/count rows must agree')
    for row in counts:
        if not isinstance(row, list) or not row:
            raise ValueError('Empty successor prior row')
        if any(not isinstance(e, list) or len(e) != 2 or type(e[0]) is not int
               or not 0 <= e[0] < bases[0] or type(e[1]) is not int or e[1] <= 0 for e in row):
            raise ValueError('Invalid successor prior code/count')
        targets = [e[0] for e in row]
        if targets != sorted(set(targets)):
            raise ValueError('Successor prior codes must be sorted and unique')


def configure_successor_prior(config, pattern_file=None, tokenizer=None, pattern_min_frequency=None):
    if getattr(config, 'adapter_itm_mode', 'default') not in ('successor_prior', 'successor_prior_learned'):
        return
    if getattr(config, 'adapter_memory_init', 'default') == 'prototype':
        raise ValueError('successor_prior does not support prototype memory initialization')
    bases = [hi-lo+1 for lo, hi in config.adapter_sid_code_ranges[:2]]
    if tokenizer is not None:
        vocab = tokenizer.get_vocab()
        for letter, (lo, hi) in zip('ab', config.adapter_sid_code_ranges):
            if any(vocab.get(f'<s_{letter}_{i}>') != lo+i for i in range(hi-lo+1)):
                raise ValueError('successor_prior requires numeric SID token ordering')
    if pattern_file is not None:
        payload = Path(pattern_file).read_bytes()
        data = json.loads(payload)
        minimum = data.get('minimum_edge_frequency')
        rows = data.get('sources')
        if (data.get('format') != 'itm_successor_prior_v1' or data.get('source_levels') != ['a','b']
                or data.get('target_level') != 'a' or bases != [data.get('sid_base')]*2
                or data.get('self_transition_policy') != 'include'
                or data.get('distribution_policy') != 'all_successors'
                or data.get('exclude_last_item') is not True
                or type(minimum) is not int or minimum < 1
                or not isinstance(rows, list) or not rows or data.get('source_count') != len(rows)):
            raise ValueError('Invalid successor prior artifact metadata')
        sources = [r['source'] for r in rows]
        if sources != sorted(sources):
            raise ValueError('Prior sources must be sorted')
        mapping = [-1]*(bases[0]*bases[1]);counts=[]
        for slot, row in enumerate(rows):
            source = row['source']
            if (len(source) != 2 or any(type(x) is not int or not 0 <= x < bases[i]
                                       for i,x in enumerate(source))):
                raise ValueError('Invalid prior source AB')
            key = source[0]*bases[1]+source[1]
            if mapping[key] >= 0:
                raise ValueError('Duplicate prior source')
            mapping[key] = slot
            successors = row.get('successors', [])
            if (not successors or any(type(e.get('count')) is not int or e['count'] <= 0 for e in successors)
                    or sum(e['count'] for e in successors) != row.get('total_count')
                    or max(e['count'] for e in successors) < minimum):
                raise ValueError('Prior total/eligibility does not match successor counts')
            counts.append([[e['a'],e['count']] for e in successors])
        for key, value in [('adapter_itm_mapping', mapping), ('adapter_itm_prior_counts', counts)]:
            previous = getattr(config, key, None)
            if previous is not None and previous != value:
                raise ValueError(f'Cannot replace checkpoint {key}')
            setattr(config, key, value)
        config.adapter_itm_pattern_manifest = {k:v for k,v in data.items() if k not in ('sources','statistics')}
        config.adapter_itm_pattern_manifest.update(artifact_sha256=hashlib.sha256(payload).hexdigest(),
                                                  threshold={'kind':'occurrences','minimum':minimum})
    validate_successor_prior(config)
    if pattern_min_frequency is not None:
        actual = (getattr(config, 'adapter_itm_pattern_manifest', None) or {}).get('threshold', {}).get('minimum')
        if pattern_min_frequency < 1 or actual != pattern_min_frequency:
            raise ValueError('Prior threshold label differs from artifact/checkpoint')


def configure_exact_itm(config, pattern_file=None, tokenizer=None, pattern_min_frequency=None):
    """Embed a validated offline artifact in config; inference needs no path."""
    if getattr(config, 'adapter_itm_mode', 'default') != 'exact':
        return
    if config.adapter_type != 'engram_structure' or config.adapter_itm_order != 1:
        raise ValueError('exact ITM requires engram_structure and adapter_itm_order=1')
    if getattr(config, 'adapter_memory_init', 'default') == 'prototype':
        raise ValueError('exact ITM does not support pre-hashed prototype initialization')
    bases = [hi - lo + 1 for lo, hi in config.adapter_sid_code_ranges[:2]]
    if tokenizer is not None:
        vocab = tokenizer.get_vocab()
        for letter, (lo, hi) in zip('ab', config.adapter_sid_code_ranges):
            if any(vocab.get(f'<s_{letter}_{i}>') != lo + i for i in range(hi - lo + 1)):
                raise ValueError('exact ITM requires SID numeric codes ordered by token ID')
    if pattern_file is not None:
        path = Path(pattern_file)
        if not path.is_file():
            raise ValueError('pattern_file must name an existing JSON file')
        payload = path.read_bytes()
        data = json.loads(payload)
        if (data.get('format') != 'exact_itm_ab_v1' or data.get('key') != 'sid_ab_local_code'
                or data.get('pattern') not in ('directed_adjacent_nonself',
                    'directed_adjacent_include_self', 'directed_adjacent_to_a')
                or data.get('base') != bases):
            raise ValueError('Pattern format or SID cardinality does not match exact ITM; use an exact artifact')
        threshold = data.get('threshold', {})
        if threshold.get('kind') != 'occurrences' or not isinstance(threshold.get('minimum'), int) or threshold['minimum'] < 1:
            raise ValueError('Invalid pattern occurrence threshold')
        mapping = data.get('mapping')
        if not isinstance(mapping, list) or len(mapping) != bases[0] * bases[1]:
            raise ValueError('Invalid pattern mapping length')
        sources = data.get('active_sources')
        if not isinstance(sources, list) or sources != sorted(sources):
            raise ValueError('Pattern sources must be stably sorted')
        expected = [-1] * len(mapping)
        for slot, pair in enumerate(sources):
            if len(pair) != 2 or any(type(x) is not int for x in pair) or not (0 <= pair[0] < bases[0] and 0 <= pair[1] < bases[1]):
                raise ValueError('Invalid source AB prefix')
            key = pair[0] * bases[1] + pair[1]
            if expected[key] >= 0:
                raise ValueError('Duplicate pattern source')
            expected[key] = slot
        if not sources or mapping != expected or data.get('source_slots') != len(sources):
            raise ValueError('Pattern mapping does not match active sources')
        existing = getattr(config, 'adapter_itm_mapping', None)
        if existing is not None and existing != mapping:
            raise ValueError('Cannot replace checkpoint exact ITM slot identities')
        config.adapter_itm_mapping = mapping
        config.adapter_itm_pattern_manifest = {k: v for k, v in data.items()
                                              if k not in ('mapping', 'edge_counts')}
        config.adapter_itm_pattern_manifest['artifact_sha256'] = hashlib.sha256(payload).hexdigest()
    if getattr(config, 'adapter_itm_mapping', None) is None:
        raise ValueError('exact ITM requires pattern_file or a checkpoint with an embedded mapping')
    if pattern_min_frequency is not None:
        manifest = getattr(config, 'adapter_itm_pattern_manifest', None) or {}
        actual = manifest.get('threshold', {}).get('minimum')
        if pattern_min_frequency < 1 or actual != pattern_min_frequency:
            raise ValueError('Pattern threshold label does not match artifact/checkpoint threshold')


class StructuredDynamicCache(DynamicCache):
    """Request-owned token context, reordered together with KV caches.

    First implementation retains token IDs and rebuilds vectorized structure once
    per forward. No hidden states or autograd graphs are retained here.
    """
    def __init__(self):
        super().__init__()
        self.structure_ids = None
        self.structure_samples = None
        self.structure_routes = None

    def _select(self, indices):
        for name in ('structure_ids', 'structure_samples', 'structure_routes'):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, indices.to(value.device)))

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self._select(beam_idx)

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._select(indices)

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        for name in ('structure_ids', 'structure_samples', 'structure_routes'):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.repeat_interleave(repeats, dim=0))

    def crop(self, max_length):
        super().crop(max_length)
        length = self.get_seq_length()
        for name in ('structure_ids', 'structure_samples', 'structure_routes'):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[:, :length])

    def batch_split(self, *args, **kwargs):
        raise NotImplementedError('Structured memory does not support split-cache/low_memory beam search')


def _shift(x, offset, fill=-1):
    if offset == 0:
        return x
    return F.pad(x, (offset, 0), value=fill)[:, :x.shape[1]]


def build_structure(input_ids, sample_idx, route, config):
    """Vectorized, prefix-only parser. Returned tensors are shared by all layers."""
    bsz, length = input_ids.shape
    valid = sample_idx >= 0
    code_count = len(config.adapter_sid_code_ranges)
    end_phase = code_count + 1
    kind = torch.full_like(input_ids, -1)
    kind = torch.where(input_ids == config.adapter_sid_begin_id, 0, kind)
    for index, (lo, hi) in enumerate(config.adapter_sid_code_ranges, 1):
        kind = torch.where((input_ids >= lo) & (input_ids <= hi), index, kind)
    kind = torch.where(input_ids == config.adapter_sid_end_id, end_phase, kind)
    phase = torch.full_like(kind, -1)
    for end in range(end_phase + 1):
        ok = valid & (kind == end)
        for offset in range(1, end + 1):
            ok = ok & (_shift(kind, offset) == end - offset)
            ok = ok & (_shift(sample_idx, offset) == sample_idx)
        phase = torch.where(ok, end, phase)
    # Commit only complete SID spans from sequence-task samples.
    positions = torch.arange(length, device=input_ids.device).expand(bsz, -1)
    complete = (phase == end_phase) & route & valid
    last = torch.where(complete, positions, -1).cummax(dim=1).values
    before = _shift(last, 1)
    previous = before.gather(1, last.clamp_min(0))
    previous = torch.where(last >= 0, previous, -1)
    histories, masks, history_positions = [], [], []
    for ends in (last, previous):
        safe = ends.clamp_min(0)
        ok = (ends >= 0) & valid & route & (sample_idx.gather(1, safe) == sample_idx)
        # Preserve the a/b item key independently of the number of SID codes.
        tokens = torch.stack([input_ids.gather(1, (ends - offset).clamp_min(0))
                              for offset in (code_count, code_count - 1)], -1)
        histories.append(tokens)
        masks.append(ok)
        # The final code (c for RecIF, d for Beauty) summarizes the complete item.
        history_positions.append((ends - 1).clamp_min(0))
    windows = {n: torch.stack([_shift(input_ids, k, 0) for k in reversed(range(n))], -1)
               for n in range(2, min(config.adapter_ngram_order, 5) + 1)}
    return dict(phase=phase, sample_idx=sample_idx, windows=windows, histories=histories,
                history_valid=masks, history_positions=history_positions,
                itm_write=(phase == 0) & route & valid)


def prepare_structure(input_ids, sample_idx, itm_sequence_mask, attention_mask, config, cache=None):
    if input_ids is None:
        raise ValueError('structured memory requires input_ids to parse SID structure')
    bsz, length = input_ids.shape
    if sample_idx is None:
        sample_idx = torch.zeros_like(input_ids)
    else:
        sample_idx = sample_idx.to(device=input_ids.device, dtype=torch.long)
        if sample_idx.shape != input_ids.shape:
            raise ValueError('sample_idx must match input_ids')
    if attention_mask is not None and attention_mask.ndim == 2:
        sample_idx = sample_idx.masked_fill(~attention_mask[:, -length:].bool(), -1)
    if itm_sequence_mask is None:
        if cache is not None and cache.structure_routes is not None:
            route = cache.structure_routes[:, -1:].expand(-1, length)
        else:
            enabled = os.environ.get('OPENONEREC_ITM_ENABLED', 'false').lower() in ('true', '1', 'yes')
            route = torch.full_like(input_ids, enabled, dtype=torch.bool)
    else:
        route = torch.as_tensor(itm_sequence_mask, device=input_ids.device, dtype=torch.bool)
        if route.ndim == 1:
            route = route[:, None]
        if route.shape == (bsz, 1):
            route = route.expand(-1, length)
        elif route.shape != input_ids.shape:
            raise ValueError('itm_sequence_mask must be [batch], [batch,1], or match input_ids')
    if cache is not None:
        if cache.structure_ids is not None:
            input_ids = torch.cat((cache.structure_ids, input_ids), dim=1)
            sample_idx = torch.cat((cache.structure_samples, sample_idx), dim=1)
            route = torch.cat((cache.structure_routes, route), dim=1)
        cache.structure_ids = input_ids.detach()
        cache.structure_samples = sample_idx.detach()
        cache.structure_routes = route.detach()
    context = build_structure(input_ids, sample_idx, route, config)
    # Decode only needs the newly supplied positions, not full-history embeddings.
    return {k: ({n: x[:, -length:] for n, x in v.items()} if isinstance(v, dict)
                else [x[:, -length:] for x in v] if isinstance(v, list)
                else v[:, -length:]) for k, v in context.items()}


class StructureMemoryBranch(nn.Module):
    def __init__(self, config, hidden_size, orders, branch="spm", memory_size=None):
        super().__init__()
        self.orders = tuple(orders)
        self.heads = config.adapter_topk
        self.size = self._next_prime(config.adapter_memory_size if memory_size is None else memory_size)
        dim = _structure_hidden_size(config, branch, hidden_size)
        self.memories = nn.ModuleList([nn.Embedding(self.size, dim) for _ in range(len(orders) * self.heads)])
        width = dim * len(orders) * self.heads
        self.key_proj = nn.Linear(width, hidden_size, bias=False)
        self.value_proj = nn.Linear(width, hidden_size, bias=False)
        self.query_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.fixed_gate = config.adapter_fixed_context_gate
        self.alpha_layer = nn.Parameter(torch.tensor([float(config.adapter_alpha_init)])) if config.adapter_use_alpha_layer else None
        self.order_salts = tuple(0x9E3779B1 + 0x85EBCA77 * order for order in self.orders)
        self.head_salts = tuple(0xC2B2AE3D + 0x27D4EB2F * head for head in range(self.heads))

    @staticmethod
    def _next_prime(value):
        def is_prime(n):
            if n < 2: return False
            if n % 2 == 0: return n == 2
            d = 3
            while d * d <= n:
                if n % d == 0: return False
                d += 2
            return True
        n = max(2, int(value))
        while not is_prime(n): n += 1
        return n

    def hash_ids(self, tokens, order_idx, head_idx):
        salt = self.order_salts[order_idx] ^ self.head_salts[head_idx]
        hashed = torch.full(tokens.shape[:-1], salt, dtype=torch.long, device=tokens.device)
        for offset in range(tokens.shape[-1]):
            token = tokens[..., offset].long() + 1
            hashed = torch.bitwise_xor(hashed * 0x9E3779B1,
                                       token * (self.head_salts[head_idx] + offset + 1))
        # abs(INT64_MIN) remains negative and would produce an invalid
        # embedding index.  Preserve the normal abs-based address mapping
        # while making that single boundary value safe.
        min_int = torch.iinfo(hashed.dtype).min
        safe_abs = torch.where(
            hashed == min_int,
            torch.full_like(hashed, min_int + 1),
            hashed.abs(),
        )
        return torch.remainder(safe_abs, self.size)

    def forward(self, hidden, keys, valid):
        pieces = []
        for oi, order in enumerate(self.orders):
            for head in range(self.heads):
                tokens = keys[oi]
                hashed = self.hash_ids(tokens, oi, head)
                value = self.memories[oi * self.heads + head](hashed)
                pieces.append(value * valid[oi].unsqueeze(-1).to(value.dtype))
        memory = torch.cat(pieces, dim=-1)
        values = self.value_proj(memory)
        if self.fixed_gate is None:
            gate = torch.sigmoid((self.query_norm(hidden) * self.key_norm(self.key_proj(memory))).sum(-1, keepdim=True) / math.sqrt(hidden.shape[-1]))
        else:
            gate = self.fixed_gate
        delta = values * gate
        return delta if self.alpha_layer is None else delta * self.alpha_layer


class ExactStructureMemoryBranch(nn.Module):
    """Compact exact lookup keyed by one completed item's SID a/b prefix."""
    def __init__(self, config, hidden_size):
        super().__init__()
        mapping = config.adapter_itm_mapping
        if not isinstance(mapping, list) or not mapping or any(type(x) is not int or x < -1 for x in mapping):
            raise ValueError('exact ITM mapping must be a flat slot lookup with -1 misses')
        slots = [x for x in mapping if x >= 0]
        active_slots = len(slots)
        if not slots or sorted(slots) != list(range(active_slots)):
            raise ValueError('exact ITM mapping must assign each compact slot exactly once')
        (self.lo_a, hi_a), (self.lo_b, hi_b) = config.adapter_sid_code_ranges[:2]
        self.base_a, self.base_b = hi_a - self.lo_a + 1, hi_b - self.lo_b + 1
        if len(mapping) != self.base_a * self.base_b:
            raise ValueError('exact ITM mapping size does not match tokenizer SID ranges')
        self.mapping = tuple(mapping)
        # Config is the source of truth. Lazy nonpersistent buffer also supports
        # meta/FSDP construction and HF loading without extra state-dict tensors.
        self.register_buffer('slot_lookup', None, persistent=False)
        dim = _structure_hidden_size(config, 'itm', hidden_size)
        self.heads = config.adapter_topk
        self.memories = nn.ModuleList([nn.Embedding(active_slots, dim) for _ in range(self.heads)])
        width = dim * self.heads
        self.next_a_head = (nn.Linear(width, self.base_a, bias=False)
                            if getattr(config, 'adapter_itm_aux_head', False) or
                            getattr(config, 'adapter_itm_aux_loss_weight', 0) > 0 else None)
        self.key_proj = nn.Linear(width, hidden_size, bias=False)
        self.value_proj = nn.Linear(width, hidden_size, bias=False)
        self.query_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.fixed_gate = config.adapter_fixed_context_gate
        self.alpha_layer = nn.Parameter(torch.tensor([float(config.adapter_alpha_init)])) if config.adapter_use_alpha_layer else None

    @property
    def active_slots(self):
        return self.memories[0].num_embeddings

    def forward(self, hidden, keys, valid, aux_labels=None):
        if self.slot_lookup is None or self.slot_lookup.device != hidden.device:
            self.slot_lookup = torch.tensor(self.mapping, dtype=torch.long, device=hidden.device)
        a, b = keys[..., 0] - self.lo_a, keys[..., 1] - self.lo_b
        in_range = (a >= 0) & (a < self.base_a) & (b >= 0) & (b < self.base_b)
        key = a * self.base_b + b
        slot = self.slot_lookup[key.clamp(min=0, max=len(self.mapping) - 1)]
        active = valid & in_range & (slot >= 0)
        memory = torch.cat([table(slot.clamp_min(0)) for table in self.memories], dim=-1)
        memory = memory * active.unsqueeze(-1).to(memory.dtype)
        values = self.value_proj(memory)
        if self.fixed_gate is None:
            gate = torch.sigmoid((self.query_norm(hidden) * self.key_norm(self.key_proj(memory))).sum(-1, keepdim=True) / math.sqrt(hidden.shape[-1]))
        else:
            gate = self.fixed_gate
        delta = values * gate
        delta = delta if self.alpha_layer is None else delta * self.alpha_layer
        if aux_labels is None:
            return delta
        selected = active & (aux_labels >= 0)
        # Empty CE with reduction=sum stays differentiable and touches each head/table.
        with torch.autocast(device_type=memory.device.type, enabled=False):
            logits = F.linear(memory[selected].float(), self.next_a_head.weight.float())
            labels = aux_labels[selected]
            loss_sum = F.cross_entropy(logits, labels, reduction='sum')
        correct = (logits.argmax(-1) == labels).sum().float()
        return delta, torch.stack((loss_sum, correct, selected.sum().float()))


class SuccessorPriorBranch(nn.Module):
    """Fixed successor probabilities, shared detached token vectors, learned gate."""
    def __init__(self, config, hidden_size):
        super().__init__()
        validate_successor_prior(config)
        self.mapping = tuple(config.adapter_itm_mapping)
        self.counts = tuple(tuple(tuple(e) for e in row) for row in config.adapter_itm_prior_counts)
        (self.lo_a, hi_a), (self.lo_b, hi_b) = config.adapter_sid_code_ranges[:2]
        self.base_a, self.base_b = hi_a-self.lo_a+1, hi_b-self.lo_b+1
        self.register_buffer('slot_lookup', None, persistent=False)
        self.register_buffer('probabilities', None, persistent=False)
        dim = _structure_hidden_size(config, 'itm', hidden_size)
        self.prior_proj = nn.Linear(hidden_size, dim, bias=False)
        self.key_proj = nn.Linear(dim, hidden_size, bias=False)
        self.value_proj = nn.Linear(dim, hidden_size, bias=False)
        self.query_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.fixed_gate = config.adapter_fixed_context_gate
        self.alpha_layer = nn.Parameter(torch.tensor([float(config.adapter_alpha_init)])) if config.adapter_use_alpha_layer else None

    def forward(self, hidden, keys, valid, token_embeddings):
        if token_embeddings is None or token_embeddings.shape != (self.base_a, hidden.shape[-1]):
            raise ValueError('successor_prior requires the shared a-token embedding matrix')
        if self.slot_lookup is None or self.slot_lookup.device != hidden.device:
            self.slot_lookup = torch.tensor(self.mapping, device=hidden.device, dtype=torch.long)
        if self.probabilities is None or self.probabilities.device != hidden.device or self.probabilities.dtype != torch.float32:
            probs = torch.zeros(len(self.counts), self.base_a, dtype=torch.float32)
            for i, row in enumerate(self.counts):
                total = sum(n for _, n in row)
                for a, n in row:probs[i,a] = n/total
            self.probabilities = probs.to(hidden.device)
        a, b = keys[...,0]-self.lo_a, keys[...,1]-self.lo_b
        slot = self.slot_lookup[(a*self.base_b+b).clamp(0,len(self.mapping)-1)]
        active = valid & (a>=0) & (a<self.base_a) & (b>=0) & (b<self.base_b) & (slot>=0)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            vectors = self.probabilities @ token_embeddings.detach().float()
        memory = vectors[slot.clamp_min(0)].to(hidden.dtype)
        memory = self.prior_proj(memory) * active.unsqueeze(-1).to(hidden.dtype)
        values = self.value_proj(memory)
        gate = (self.fixed_gate if self.fixed_gate is not None else torch.sigmoid(
            (self.query_norm(hidden)*self.key_norm(self.key_proj(memory))).sum(-1,keepdim=True)/math.sqrt(hidden.shape[-1])))
        delta = values*gate
        return delta if self.alpha_layer is None else delta*self.alpha_layer


class SuccessorPriorLearnedBranch(nn.Module):
    """Successor prior using learned per-a tables instead of LM embeddings."""
    def __init__(self, config, hidden_size):
        super().__init__()
        validate_successor_prior(config)
        self.mapping = tuple(config.adapter_itm_mapping)
        self.counts = tuple(tuple(tuple(e) for e in row) for row in config.adapter_itm_prior_counts)
        (self.lo_a, hi_a), (self.lo_b, hi_b) = config.adapter_sid_code_ranges[:2]
        self.base_a, self.base_b = hi_a-self.lo_a+1, hi_b-self.lo_b+1
        self.register_buffer('slot_lookup', None, persistent=False)
        self.register_buffer('probabilities', None, persistent=False)
        dim = _structure_hidden_size(config, 'itm', hidden_size)
        self.heads = config.adapter_topk
        self.memories = nn.ModuleList([nn.Embedding(self.base_a, dim) for _ in range(self.heads)])
        width = self.heads * dim
        self.key_proj = nn.Linear(width, hidden_size, bias=False)
        self.value_proj = nn.Linear(width, hidden_size, bias=False)
        self.query_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.fixed_gate = config.adapter_fixed_context_gate
        self.alpha_layer = nn.Parameter(torch.tensor([float(config.adapter_alpha_init)])) if config.adapter_use_alpha_layer else None

    def _probabilities(self, device):
        if self.probabilities is None or self.probabilities.device != device:
            probs = torch.zeros(len(self.counts), self.base_a, dtype=torch.float32)
            for i, row in enumerate(self.counts):
                total = sum(n for _, n in row)
                for a, n in row:
                    probs[i, a] = n / total
            self.probabilities = probs.to(device)
        return self.probabilities

    def forward(self, hidden, keys, valid):
        if self.slot_lookup is None or self.slot_lookup.device != hidden.device:
            self.slot_lookup = torch.tensor(self.mapping, dtype=torch.long, device=hidden.device)
        probabilities = self._probabilities(hidden.device)
        a, b = keys[..., 0] - self.lo_a, keys[..., 1] - self.lo_b
        slot = self.slot_lookup[(a * self.base_b + b).clamp(0, len(self.mapping) - 1)]
        active = valid & (a >= 0) & (a < self.base_a) & (b >= 0) & (b < self.base_b) & (slot >= 0)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            vectors = [torch.matmul(probabilities, table.weight.float()) for table in self.memories]
        memory = torch.cat([vectors[h][slot.clamp_min(0)] for h in range(self.heads)], dim=-1).to(hidden.dtype)
        memory = memory * active.unsqueeze(-1).to(memory.dtype)
        values = self.value_proj(memory)
        if self.fixed_gate is None:
            gate = torch.sigmoid((self.query_norm(hidden) * self.key_norm(self.key_proj(memory))).sum(-1, keepdim=True) / math.sqrt(hidden.shape[-1]))
        else:
            gate = self.fixed_gate
        delta = values * gate
        return delta if self.alpha_layer is None else delta * self.alpha_layer


class StructuredEngramMemoryAdapter(BaseMemoryAdapter):
    def __init__(self, config, hidden_size):
        super().__init__(hidden_size)
        defaults = dict(adapter_use_spm=True, adapter_use_itm=True, adapter_itm_order=1,
                        adapter_ngram_order=3, adapter_topk=4, adapter_memory_size=1024,
                        adapter_spm_memory_size=None, adapter_itm_memory_size=None,
                        adapter_hidden_size=None, adapter_spm_hidden_size=None,
                        adapter_itm_hidden_size=None, adapter_fixed_context_gate=None,
                        adapter_use_alpha_layer=False, adapter_alpha_init=.1,
                        adapter_sid_begin_id=None, adapter_sid_end_id=None, adapter_sid_code_ranges=None,
                        adapter_itm_mode='default', adapter_itm_mapping=None,
                        adapter_itm_prior_counts=None)
        for key, value in defaults.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if not (config.adapter_use_spm or config.adapter_use_itm):
            raise ValueError('engram_structure requires SPM and/or ITM')
        common_dim = config.adapter_hidden_size or hidden_size
        if ((config.adapter_spm_hidden_size is not None and config.adapter_spm_hidden_size < 1) or
                (config.adapter_itm_hidden_size is not None and config.adapter_itm_hidden_size < 1)):
            raise ValueError('Structure memory hidden sizes must be positive')
        config.adapter_spm_hidden_size = (config.adapter_spm_hidden_size or common_dim)
        config.adapter_itm_hidden_size = (config.adapter_itm_hidden_size or common_dim)
        if config.adapter_spm_hidden_size < 1 or config.adapter_itm_hidden_size < 1:
            raise ValueError('Structure memory hidden sizes must be positive')
        if not 2 <= config.adapter_ngram_order <= 5:
            raise ValueError('SPM ngram order must be between 2 and 5')
        if config.adapter_itm_order not in (1, 2):
            raise ValueError('ITM order must be 1 or 2')
        if config.adapter_itm_mode not in ('default', 'exact', 'successor_prior', 'successor_prior_learned'):
            raise ValueError('Invalid adapter_itm_mode')
        if config.adapter_itm_mode in ('exact', 'successor_prior', 'successor_prior_learned') and config.adapter_use_itm:
            if config.adapter_itm_order != 1:
                raise ValueError('exact ITM uses only the latest completed item; set adapter_itm_order=1')
            if config.adapter_itm_mapping is None:
                raise ValueError('exact ITM requires adapter_itm_mapping from a pattern artifact')
        self.spm_memory_size = (config.adapter_memory_size if config.adapter_spm_memory_size is None
                                else config.adapter_spm_memory_size)
        self.itm_memory_size = (config.adapter_memory_size if config.adapter_itm_memory_size is None
                                else config.adapter_itm_memory_size)
        if (config.adapter_memory_size < 2 or self.spm_memory_size < 2 or
                self.itm_memory_size < 2 or config.adapter_topk < 1):
            raise ValueError('Invalid structure memory capacity/head count')
        # Store effective values so converted configs and diagnostics are explicit.
        config.adapter_spm_memory_size = self.spm_memory_size
        config.adapter_itm_memory_size = self.itm_memory_size
        if config.adapter_fixed_context_gate is not None and not 0 <= config.adapter_fixed_context_gate <= 1:
            raise ValueError('Context gate must be in [0,1]')
        if config.adapter_sid_begin_id is None or config.adapter_sid_end_id is None or not config.adapter_sid_code_ranges or len(config.adapter_sid_code_ranges) < 2:
            raise ValueError('engram_structure requires tokenizer-derived SID IDs/ranges in config')
        self.spm = (StructureMemoryBranch(config, hidden_size, range(2, config.adapter_ngram_order + 1),
                                          "spm", self.spm_memory_size)
                    if config.adapter_use_spm else None)
        if not config.adapter_use_itm:
            self.itm = None
        elif config.adapter_itm_mode == 'exact':
            self.itm = ExactStructureMemoryBranch(config, hidden_size)
        elif config.adapter_itm_mode == 'successor_prior':
            self.itm = SuccessorPriorBranch(config, hidden_size)
        elif config.adapter_itm_mode == 'successor_prior_learned':
            self.itm = SuccessorPriorLearnedBranch(config, hidden_size)
        else:
            self.itm = StructureMemoryBranch(config, hidden_size, range(1, config.adapter_itm_order + 1),
                                             "itm", self.itm_memory_size)

    def forward(self, hidden_states, input_ids=None, attention_mask=None, layer_idx=None,
                structure_context=None, lookup_ids=None, itm_aux_labels=None, **kwargs):
        if structure_context is None:
            raise ValueError('Missing shared structure_context')
        c = structure_context
        delta = torch.zeros_like(hidden_states)
        if self.spm is not None:
            keys = [c['windows'][n] for n in self.spm.orders]
            masks = [c['phase'] >= n - 1 for n in self.spm.orders]
            delta = delta + self.spm(hidden_states, keys, masks)
        if self.itm is not None:
            if isinstance(self.itm, SuccessorPriorBranch):
                return delta + self.itm(hidden_states, c['histories'][0],
                                       c['history_valid'][0] & c['itm_write'], kwargs.get('itm_token_embeddings'))
            if isinstance(self.itm, SuccessorPriorLearnedBranch):
                return delta + self.itm(hidden_states, c['histories'][0],
                                       c['history_valid'][0] & c['itm_write'])
            if isinstance(self.itm, ExactStructureMemoryBranch):
                result = self.itm(hidden_states, c['histories'][0],
                                  c['history_valid'][0] & c['itm_write'], itm_aux_labels)
                if itm_aux_labels is not None:
                    itm_delta, stats = result
                    return delta + itm_delta, stats
                return delta + result
            keys = [c['histories'][0]]
            masks = [c['history_valid'][0] & c['itm_write']]
            if len(self.itm.orders) == 2:
                keys.append(torch.cat((c['histories'][1], c['histories'][0]), -1))
                masks.append(c['history_valid'][1] & c['history_valid'][0] & c['itm_write'])
            delta = delta + self.itm(hidden_states, keys, masks)
        return delta
