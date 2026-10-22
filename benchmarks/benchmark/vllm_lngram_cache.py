"""Evaluation-only latent-symbol cache, addressed by native vLLM KV slots.

No parameters or checkpoint files are changed. The worker mirrors KV block
copy/swap operations so prefix reuse and branching retain the same symbols.
"""
from types import MethodType

import torch


class SymbolCache:
    def __init__(self, branch):
        self.routes = branch.routes
        self.dtype = (torch.uint8 if branch.bits <= 8 else
                      torch.int16 if branch.bits <= 15 else
                      torch.int32 if branch.bits <= 31 else torch.int64)
        self.gpu = self.cpu = None
        self.gpu_valid = self.cpu_valid = None

    def block_bytes(self, block_size):
        return block_size * (self.routes * torch.empty((), dtype=self.dtype).element_size() + 1)

    def allocate(self, gpu_blocks, cpu_blocks, block_size, device):
        self.gpu = torch.zeros((gpu_blocks, block_size, self.routes), dtype=self.dtype, device=device)
        self.cpu = torch.zeros((cpu_blocks, block_size, self.routes), dtype=self.dtype)
        self.gpu_valid = torch.zeros((gpu_blocks, block_size), dtype=torch.bool, device=device)
        self.cpu_valid = torch.zeros((cpu_blocks, block_size), dtype=torch.bool)

    def write(self, slots, symbols):
        if self.gpu is None:
            raise RuntimeError('Lngram symbol cache was not initialized by StructureWorker')
        slots = slots.reshape(-1)
        self.gpu.view(-1, self.routes).index_copy_(0, slots, symbols.reshape(-1, self.routes).to(self.dtype))
        self.gpu_valid.view(-1).index_fill_(0, slots, True)

    def read(self, slots, mask, dtype):
        # Invalid keys are arbitrary; never dereference them.
        valid = mask
        while valid.ndim < slots.ndim:
            valid = valid.unsqueeze(-1)
        slots = torch.where(valid, slots, 0)
        populated = self.gpu_valid.view(-1)[slots]
        # Do not use torch._assert_async here.  A failed asynchronous device
        # assertion poisons the CUDA context and makes the next visible error
        # appear in an unrelated Qwen rotary kernel.  Validate the slot map
        # synchronously before dereferencing the cache.
        if not bool(torch.all(populated | ~valid).item()):
            raise RuntimeError('Missing Lngram history in native KV slot')
        return self.gpu.view(-1, self.routes)[slots].to(dtype)

    def copy(self, mapping):
        if mapping is None or not mapping.numel():
            return
        source, target = mapping.to(self.gpu.device).unbind(-1)
        # Snapshot all sources before writing, including one-to-many COW.
        self.gpu.index_copy_(0, target, self.gpu.index_select(0, source))
        self.gpu_valid.index_copy_(0, target, self.gpu_valid.index_select(0, source))

    def swap(self, mapping, swap_in):
        if mapping is None or not mapping.numel():
            return
        source, target = (self.cpu, self.gpu) if swap_in else (self.gpu, self.cpu)
        source_valid, target_valid = ((self.cpu_valid, self.gpu_valid) if swap_in
                                      else (self.gpu_valid, self.cpu_valid))
        # Swap mappings are CPU tensors in vLLM 0.7.3. Blocking copies keep
        # ordering explicit; swapping is only used under memory pressure.
        for src, dst in mapping.tolist():
            target[dst].copy_(source[src])
            target_valid[dst].copy_(source_valid[src])


def cached_lngram_forward(self, hidden_states, input_ids=None, attention_mask=None,
                          layer_idx=None, structure_context=None, **kwargs):
    c = structure_context
    if c is None or 'symbol_slots' not in c:
        # Engine memory profiling has dummy tokens and no allocated KV slots.
        if any(cache.gpu is not None for cache in self._vllm_symbol_caches.values()):
            raise RuntimeError('Missing Lngram slot context during live inference')
        return self._vllm_original_forward(
            hidden_states, input_ids=input_ids, attention_mask=attention_mask,
            layer_idx=layer_idx, structure_context=c, **kwargs)
    delta = torch.zeros_like(hidden_states)
    branches = ('spm', 'itm') if hasattr(self, 'spm') else ('memory',)
    for name in branches:
        branch = getattr(self, name) if name != 'memory' else self.memory
        if branch is None:
            continue
        cache = self._vllm_symbol_caches[name]
        symbols = branch.symbols_from_logits(branch.route_logits(hidden_states))
        cache.write(c['symbol_slots'], symbols)
        windows, masks = [], []
        for order in branch.orders:
            if name in ('spm', 'memory'):
                # Both ordinary Lngram and structured SPM use scheduler-built
                # windows over the complete request history.  The current
                # query slots alone are insufficient during decode because
                # their left neighbors live in earlier KV blocks.
                mask = c['phase'] >= order - 1
                slots = c['spm_slots'][order]
            else:
                mask = c['itm_write']
                for index in range(order):
                    mask = mask & c['history_valid'][index]
                slots = torch.stack(list(reversed(c['itm_slots'][:order])), -1)
            windows.append(cache.read(slots, mask, symbols.dtype))
            masks.append(mask)
        delta += branch.inject(hidden_states, branch.lookup(windows, masks))
    return delta


def install_symbol_caches(model):
    caches = []
    for module in model.modules():
        if module.__class__.__name__ not in ('StructuredLngramMemoryAdapter', 'LngramMemoryAdapter'):
            continue
        if hasattr(module, '_vllm_symbol_caches'):
            raise RuntimeError('Lngram cache installed twice')
        if hasattr(module, 'spm'):
            branches = ('spm', 'itm')
        elif hasattr(module, 'memory'):
            branches = ('memory',)
        else:
            branches = ()
        module._vllm_symbol_caches = {
            name: SymbolCache(getattr(module, name) if name != 'memory' else module.memory)
            for name in branches if getattr(module, name, module.memory if name == 'memory' else None) is not None}
        caches.extend(module._vllm_symbol_caches.values())
        module._vllm_original_forward = module.forward
        module.forward = MethodType(cached_lngram_forward, module)
    if not caches:
        raise ValueError('Lngram export has no supported Lngram adapters')
    return caches
