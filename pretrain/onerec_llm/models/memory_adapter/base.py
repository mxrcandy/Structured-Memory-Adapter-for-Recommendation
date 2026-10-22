from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class BaseMemoryAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class NullMemoryAdapter(BaseMemoryAdapter):
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def build_memory_adapter(config, hidden_size: int) -> BaseMemoryAdapter:
    adapter_type = getattr(config, "adapter_type", None)
    if getattr(config, "adapter_itm_mode", "default") == "exact" and adapter_type != "engram_structure":
        raise ValueError('exact ITM is supported only for engram_structure')
    adapter_hidden_size = getattr(config, "adapter_hidden_size", None) or hidden_size
    memory_size = getattr(config, "adapter_memory_size", 1024)
    topk = getattr(config, "adapter_topk", 4)
    if adapter_type in (None, "", "none"):
        return NullMemoryAdapter(hidden_size)
    if adapter_type == "engram_structure":
        from .structure import StructuredEngramMemoryAdapter

        return StructuredEngramMemoryAdapter(config, hidden_size)
    if adapter_type == "lngram_structure":
        from .lngram_structure import StructuredLngramMemoryAdapter
        return StructuredLngramMemoryAdapter(config, hidden_size)
    if adapter_type == "lngram":
        from .lngram import LngramMemoryAdapter
        return LngramMemoryAdapter(config, hidden_size)
    if adapter_type == "engram":
        from .engram import EngramMemoryAdapter

        return EngramMemoryAdapter(
            hidden_size=hidden_size,
            adapter_hidden_size=adapter_hidden_size,
            memory_size=memory_size,
            ngram_order=getattr(config, "adapter_ngram_order", 2),
            topk=topk,
            hash_heads=topk,
            use_alpha_layer=getattr(config, "adapter_use_alpha_layer", False),
            alpha_init=getattr(config, "adapter_alpha_init", 0.1),
            fixed_context_gate=getattr(config, "adapter_fixed_context_gate", None),
            use_conv=getattr(config, "adapter_use_conv", True),
            use_sid_span=getattr(config, "adapter_use_sid_span", False),
            sid_token_start=getattr(config, "adapter_sid_token_start", None),
            sid_token_end=getattr(config, "adapter_sid_token_end", None),
        )
    raise ValueError(
        f"Unsupported adapter_type: {adapter_type}. Supported: 'engram', 'engram_structure', "
        "'lngram', 'lngram_structure'."
    )
