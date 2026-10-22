from __future__ import annotations

import torch

from .base import BaseMemoryAdapter
from .lngram import LatentSymbolMemory, _LatentLookupSurrogate


def _shift_symbols(symbols, offset):
    if offset == 0:
        return symbols
    return torch.nn.functional.pad(symbols, (0, 0, offset, 0))[:, :symbols.shape[1]]


class StructuredLngramMemoryAdapter(BaseMemoryAdapter):
    def __init__(self, config, hidden_size: int):
        super().__init__(hidden_size)
        if not (config.adapter_use_spm or config.adapter_use_itm):
            raise ValueError("lngram_structure requires SPM and/or ITM")
        self.spm = (LatentSymbolMemory(config, hidden_size, range(2, config.adapter_ngram_order + 1))
                    if config.adapter_use_spm else None)
        self.itm = (LatentSymbolMemory(config, hidden_size, range(1, config.adapter_itm_order + 1))
                    if config.adapter_use_itm else None)

    def forward(self, hidden_states, input_ids=None, attention_mask=None, layer_idx=None,
                structure_context=None, **kwargs):
        if structure_context is None:
            raise ValueError("lngram_structure requires structure_context")
        c = structure_context
        delta = torch.zeros_like(hidden_states)
        if self.spm is not None:
            route_logits = self.spm.route_logits(hidden_states)
            symbols = self.spm.symbols_from_logits(route_logits)
            windows, masks = [], []
            for order in self.spm.orders:
                windows.append(torch.stack([_shift_symbols(symbols, k)
                                            for k in reversed(range(order))], 2))
                masks.append(c["phase"] >= order - 1)
            memory = self.spm.lookup(windows, masks)
            memory = _LatentLookupSurrogate.apply(memory, route_logits, symbols, self.spm)
            delta += self.spm.inject(hidden_states, memory)
        if self.itm is not None:
            route_logits = self.itm.route_logits(hidden_states)
            symbols = self.itm.symbols_from_logits(route_logits)
            histories, masks = [], []
            for order in self.itm.orders:
                # During incremental decode the structure parser may refer to
                # positions in the retained token history; the current hidden
                # chunk only contains query states, so clamp as a conservative
                # fallback until hidden-state history caching is added.
                positions = [p.clamp_max(hidden_states.shape[1] - 1) for p in c["history_positions"][:order]]
                gathered = [symbols.gather(1, p.clamp_min(0).unsqueeze(-1).expand(-1, -1, symbols.shape[-1]))
                            for p in reversed(positions)]
                histories.append(torch.stack(gathered, 2))
                ok = c["itm_write"]
                for index in range(order):
                    ok = ok & c["history_valid"][index]
                masks.append(ok)
            memory = self.itm.lookup(histories, masks)
            memory = _LatentLookupSurrogate.apply(memory, route_logits, symbols, self.itm)
            delta += self.itm.inject(hidden_states, memory)
        return delta
