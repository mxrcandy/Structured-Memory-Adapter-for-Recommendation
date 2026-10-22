"""SPM/ITM bridge for the same vLLM Transformers backend used by Engram.

vLLM still owns scheduling, attention, KV blocks, beam search and sampling.
This module only transports causal structure from scheduler-owned histories
to the exported Qwen3 model. It does not implement another backbone.
"""
import dataclasses
from pathlib import Path
import sys

import torch
from vllm import ModelRegistry
from vllm.model_executor.models.transformers import TransformersModel
from vllm.worker.worker import Worker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pretrain'))
from onerec_llm.models.memory_adapter.structure import build_structure

ARCHITECTURE = 'OpenOneRecStructuredTransformersModel'


def _next_prime(value):
    value = max(2, int(value))
    while any(value % d == 0 for d in range(2, int(value ** 0.5) + 1)):
        value += 1
    return value


def engram_lookup_ids(ids, config):
    """Build legacy Engram addresses from the scheduler's complete history."""
    memory_size = _next_prime(getattr(config, 'adapter_memory_size', 1024))
    order_max = max(2, int(getattr(config, 'adapter_ngram_order', 2)))
    heads = max(1, int(getattr(config, 'adapter_topk', 4)))
    orders = tuple(range(2, order_max + 1))
    order_salts = tuple(0x9E3779B1 + 0x85EBCA77 * order for order in orders)
    head_salts = tuple(0xC2B2AE3D + 0x27D4EB2F * head for head in range(heads))
    outputs = []
    for order_idx, order in enumerate(orders):
        padded = torch.nn.functional.pad(ids, (order - 1, 0), value=0)
        windows = padded.unfold(1, order, 1)
        for head_idx in range(heads):
            hashed = torch.full(windows.shape[:-1], order_salts[order_idx] ^ head_salts[head_idx], dtype=torch.long, device=ids.device)
            for offset in range(order):
                token = windows[..., offset].long() + 1
                hashed = torch.bitwise_xor(hashed * 0x9E3779B1,
                                           token * (head_salts[head_idx] + offset + 1))
            # Preserve the trained abs-based address mapping and handle the
            # sole value for which abs() remains negative.
            min_int = torch.iinfo(hashed.dtype).min
            safe_abs = torch.where(hashed == min_int,
                                   torch.full_like(hashed, min_int + 1),
                                   hashed.abs())
            outputs.append(torch.remainder(safe_abs, memory_size))
    return torch.stack(outputs, dim=-1)


def register_structure_model():
    import vllm
    if vllm.__version__ != '0.7.3':
        raise RuntimeError('Structured vLLM bridge currently requires tested vLLM 0.7.3')
    if ARCHITECTURE in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCHITECTURE, 'benchmark.vllm_structure:StructuredTransformersModel')


def context_to_kwargs(context):
    result = {}
    if 'phase' in context:
        result.update(structure_phase=context['phase'], structure_itm_write=context['itm_write'])
        result.update({f'structure_window_{n}': t for n, t in context['windows'].items()})
        for i in range(2):
            result[f'structure_history_{i}'] = context['histories'][i]
            result[f'structure_valid_{i}'] = context['history_valid'][i]
            result[f'structure_position_{i}'] = context['history_positions'][i]
    if 'symbol_slots' in context and context['symbol_slots'] is not None:
        result['structure_symbol_slots'] = context['symbol_slots']
        if 'spm_slots' in context:
            result.update({f'structure_spm_slots_{n}': t for n, t in context['spm_slots'].items()})
        if 'itm_slots' in context:
            for i in range(2):
                result[f'structure_itm_slots_{i}'] = context['itm_slots'][i]
    if 'memory_lookup_ids' in context:
        result['vllm_memory_lookup_ids'] = context['memory_lookup_ids']
    return result


def kwargs_to_context(kwargs, config):
    context = dict(phase=kwargs['structure_phase'], itm_write=kwargs['structure_itm_write'],
                windows={n: kwargs[f'structure_window_{n}'] for n in range(2, config.adapter_ngram_order + 1)},
                histories=[kwargs[f'structure_history_{i}'] for i in range(2)],
                history_valid=[kwargs[f'structure_valid_{i}'] for i in range(2)],
                history_positions=[kwargs[f'structure_position_{i}'] for i in range(2)])
    if 'structure_symbol_slots' in kwargs:
        context.update(symbol_slots=kwargs['structure_symbol_slots'],
                       spm_slots={n: kwargs[f'structure_spm_slots_{n}']
                                  for n in range(2, config.adapter_ngram_order + 1)})
        if 'structure_itm_slots_0' in kwargs:
            context['itm_slots'] = [kwargs[f'structure_itm_slots_{i}'] for i in range(2)]
    return context


def scheduler_context(histories, seq_lens, query_lens, config, route, device,
                      block_tables=None, block_size=None):
    """Parse once across layers, retain only the current queries in the result.

    Distinct sample IDs allow concatenation without longest-request padding.
    Only histories up to each query end are included (also for chunked prefill).
    """
    if not (len(histories) == len(seq_lens) == len(query_lens)):
        raise ValueError('Scheduler sequence metadata is inconsistent')
    if block_tables is not None and (len(block_tables) != len(histories) or not block_size or block_size < 1):
        raise ValueError('Invalid Lngram block table metadata')
    # Older converted configs (and CPU callers that exercise the bridge
    # directly) may not carry adapter_type.  Presence of KV block metadata
    # identifies the latent-symbol path; otherwise retain the structured
    # token-context behavior used by the legacy bridge.
    adapter_type = getattr(
        config, 'adapter_type',
        'lngram_structure' if block_tables is not None else 'engram_structure')
    tokens, samples, selected, slots, phases = [], [], [], [], []
    for sample, (history, length, query) in enumerate(zip(histories, seq_lens, query_lens)):
        if not 0 < query <= length <= len(history):
            raise ValueError('Incomplete scheduler history for structured memory')
        offset = len(tokens)
        tokens.extend(history[:length])
        samples.extend([sample] * length)
        phases.extend(range(length))
        selected.extend(range(offset + length - query, offset + length))
        if block_tables is not None:
            table = block_tables[sample]
            if len(table) * block_size < length or any(block < 0 for block in table):
                raise ValueError('Incomplete Lngram KV block table')
            slots.extend(table[pos // block_size] * block_size + pos % block_size
                         for pos in range(length))
    ids = torch.tensor(tokens, dtype=torch.long, device=device)[None]
    sample_ids = torch.tensor(samples, dtype=torch.long, device=device)[None]
    eligible = torch.full_like(ids, route, dtype=torch.bool)
    context = build_structure(ids, sample_ids, eligible, config) if adapter_type in ('engram_structure', 'lngram_structure') else {}
    if adapter_type in ('engram', 'lngram'):
        phase = torch.tensor(phases, dtype=torch.long, device=device)[None]
        zeros = torch.zeros_like(ids)
        context.update(phase=phase, itm_write=torch.zeros_like(ids, dtype=torch.bool),
                       windows={n: torch.zeros((1, ids.shape[1], n), dtype=torch.long, device=device)
                                for n in range(2, config.adapter_ngram_order + 1)},
                       histories=[torch.zeros((1, ids.shape[1], 2), dtype=torch.long, device=device) for _ in range(2)],
                       history_valid=[torch.zeros_like(ids, dtype=torch.bool) for _ in range(2)],
                       history_positions=[zeros for _ in range(2)])
    if adapter_type in ('engram', 'engram_structure'):
        context['memory_lookup_ids'] = engram_lookup_ids(ids, config)
    if block_tables is not None:
        full_slots = torch.tensor(slots, dtype=torch.long, device=device)[None]
        context['symbol_slots'] = full_slots
        context['spm_slots'] = {
            n: torch.stack([torch.nn.functional.pad(full_slots, (k, 0))[:, :ids.shape[1]]
                            for k in reversed(range(n))], -1)
            for n in range(2, config.adapter_ngram_order + 1)}
        context['itm_slots'] = [full_slots.gather(1, pos) for pos in context['history_positions']]
    indices = torch.tensor(selected, dtype=torch.long, device=device)
    return {key: tensor[0].index_select(0, indices)
            for key, tensor in context_to_kwargs(context).items()}


class StructureRunner:
    def __init__(self, runner):
        self.runner = runner
        self.itm_route = False

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def prepare_model_input(self, seq_group_metadata_list, *args, **kwargs):
        model_input = self.runner.prepare_model_input(seq_group_metadata_list, *args, **kwargs)
        if not seq_group_metadata_list:
            return model_input
        histories = [list(data.get_token_ids()) for group in seq_group_metadata_list
                     for data in group.seq_data.values()]
        slot_kwargs = {}
        if self.model_config.hf_config.adapter_type in ('lngram', 'lngram_structure'):
            slot_kwargs = dict(
                block_tables=[group.block_tables[seq_id] for group in seq_group_metadata_list
                              for seq_id in group.seq_data],
                block_size=self.runner.block_size)
        extra = scheduler_context(histories, model_input.seq_lens, model_input.query_lens,
                                  self.model_config.hf_config, self.itm_route, self.device, **slot_kwargs)
        if 'structure_phase' in extra and extra['structure_phase'].shape != model_input.input_tokens.shape:
            raise ValueError('Structured query length disagrees with native model input')
        if 'structure_symbol_slots' in extra:
            native_slots = model_input.attn_metadata.slot_mapping
            if extra['structure_symbol_slots'].shape != native_slots.shape:
                raise ValueError(
                    'Lngram slot shape disagrees with native vLLM attention slots: '
                    f'{tuple(extra["structure_symbol_slots"].shape)} != {tuple(native_slots.shape)}')
            # This check must be synchronous.  An async device assertion can
            # poison the CUDA context and make the next visible failure look
            # like an unrelated rotary-embedding illegal access.
            if not bool(torch.equal(extra['structure_symbol_slots'], native_slots)):
                raise ValueError('Lngram slots disagree with native vLLM attention slots')
        if model_input.multi_modal_kwargs:
            raise ValueError('Structured memory currently supports text-only requests')
        # Reuse the existing tensor-kwargs transport, including TP broadcast.
        # These internal tensors do not make the model a multimodal model.
        return dataclasses.replace(model_input, multi_modal_kwargs=extra)


class StructureWorker(Worker):
    def __init__(self, *args, **kwargs):
        register_structure_model()
        kwargs['model_runner_cls'] = StructureRunner
        super().__init__(*args, **kwargs)

    def get_cache_block_size_bytes(self):
        # Include symbols when vLLM budgets both GPU blocks and CPU swap space.
        return super().get_cache_block_size_bytes() + sum(
            cache.block_bytes(self.cache_config.block_size)
            for cache in self.model_runner.model.symbol_caches)

    def _init_cache_engine(self):
        super()._init_cache_engine()
        for cache in self.model_runner.model.symbol_caches:
            cache.allocate(self.cache_config.num_gpu_blocks, self.cache_config.num_cpu_blocks,
                           self.cache_config.block_size, self.device)

    def execute_worker(self, worker_input):
        super().execute_worker(worker_input)
        for cache in self.model_runner.model.symbol_caches:
            cache.swap(worker_input.blocks_to_swap_in, swap_in=True)
            cache.swap(worker_input.blocks_to_swap_out, swap_in=False)
            cache.copy(worker_input.blocks_to_copy)

    def set_structure_route(self, enabled):
        self.model_runner.itm_route = bool(enabled)


class StructuredTransformersModel(TransformersModel):
    """Keep the old backend's constructor, attention and weight loading."""
    def __init__(self, *, vllm_config, prefix=''):
        c = vllm_config.model_config.hf_config
        if c.adapter_type not in ('engram', 'lngram', 'engram_structure', 'lngram_structure'):
            raise ValueError('Unsupported memory adapter type for unified bridge')
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError('Structured bridge requires pipeline_parallel_size=1')
        if not vllm_config.model_config.enforce_eager:
            raise ValueError('Structured bridge currently requires enforce_eager=True')
        if vllm_config.scheduler_config.is_multi_step or vllm_config.speculative_config is not None:
            raise ValueError('Multi-step/speculative decoding is not supported')
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None:
            raise ValueError('Quantization/LoRA is not supported by the structured bridge')
        if getattr(c, 'use_sliding_window', False):
            raise ValueError('Sliding-window structured decoding is not supported')
        if getattr(vllm_config.model_config, 'enable_sleep_mode', False):
            raise ValueError('Structured symbol caches do not support vLLM sleep mode')
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not getattr(self.model, 'supports_vllm_structure_context', False):
            raise ValueError('Re-convert checkpoint: exported Qwen3 lacks the structure-context bridge')
        self.symbol_caches = []
        if c.adapter_type in ('lngram', 'lngram_structure'):
            from benchmark.vllm_lngram_cache import install_symbol_caches
            self.symbol_caches = install_symbol_caches(self.model)

    def forward(self, input_ids, positions, kv_caches, attn_metadata,
                intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError('Structured bridge requires ordinary text token input')
        if input_ids.ndim != 1 or positions.ndim != 1 or input_ids.numel() != positions.numel():
            raise ValueError(
                'Native vLLM query tensors must be flattened and aligned: '
                f'input_ids={tuple(input_ids.shape)}, positions={tuple(positions.shape)}')
        if positions.numel() and (
                int(positions.min().item()) < 0 or
                int(positions.max().item()) >= int(self.config.max_position_embeddings)):
            raise ValueError(
                'Native vLLM position is outside the rotary range: '
                f'min={int(positions.min().item())}, max={int(positions.max().item())}, '
                f'max_position_embeddings={self.config.max_position_embeddings}')
        lookup_ids = kwargs.get('vllm_memory_lookup_ids')
        if lookup_ids is not None and lookup_ids.ndim == 2:
            lookup_ids = lookup_ids[None]
        if 'structure_phase' in kwargs:
            context = kwargs_to_context({key: value[None] for key, value in kwargs.items()}, self.config)
        else:
            if any(cache.numel() for cache in kv_caches):
                raise RuntimeError('Missing scheduler memory context; use StructureWorker')
            context = None
        if context is None and self.config.adapter_type in ('engram_structure', 'lngram_structure'):
            context = build_structure(input_ids[None], torch.zeros_like(input_ids)[None],
                                      torch.zeros_like(input_ids, dtype=torch.bool)[None], self.config)
        return self.model(input_ids[None], use_cache=False, position_ids=positions[None],
                          attn_metadata=attn_metadata, attention_instances=self.attention_instances,
                          **({'vllm_memory_lookup_ids': lookup_ids} if lookup_ids is not None else {}),
                          **({'vllm_structure_context': context} if context is not None else {}),
                          return_dict=False)[0][0]


def native_kwargs():
    register_structure_model()
    return dict(hf_overrides={'architectures': [ARCHITECTURE]},
                worker_cls='benchmark.vllm_structure.StructureWorker',
                enforce_eager=True, disable_async_output_proc=True)


def set_itm_route(llm, enabled):
    """Benchmark calls are task-homogeneous and finish before route changes."""
    enabled = bool(enabled)
    if getattr(llm, '_structure_itm_route', None) == enabled:
        return
    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('Cannot change ITM routing while requests are running')
    # Prefix KV values depend on routing; do not reuse a previous task's cache.
    reset = llm.reset_prefix_cache()
    if llm.llm_engine.cache_config.enable_prefix_caching and not reset:
        raise RuntimeError('Cannot change ITM route: prefix cache reset failed')
    llm.llm_engine.model_executor.collective_rpc('set_structure_route', args=(enabled,))
    llm._structure_itm_route = enabled
