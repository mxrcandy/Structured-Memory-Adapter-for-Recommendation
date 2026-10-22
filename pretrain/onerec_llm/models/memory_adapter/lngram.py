from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseMemoryAdapter


class _LatentLookupSurrogate(torch.autograd.Function):
    """Identity forward with a local bit-flip gradient for discrete routes."""

    @staticmethod
    def forward(ctx, memory, route_logits, route_codes, module):
        ctx.module = module
        ctx.save_for_backward(route_logits, route_codes)
        return memory

    @staticmethod
    def backward(ctx, grad_memory):
        route_logits, route_codes = ctx.saved_tensors
        module = ctx.module
        grad_logits = torch.zeros_like(route_logits)
        batch, length, num_orders, routes, _ = grad_memory.shape
        bit_values = (2 ** torch.arange(module.bits, device=route_codes.device)).long()
        route_ids = torch.arange(routes, device=route_codes.device).view(1, routes)
        for order_index, order in enumerate(module.orders):
            if length < order:
                continue
            # A route symbol at position p participates in outputs p..p+order-1.
            # Keep the exact bit-flip surrogate, but evaluate all output
            # positions/route bits in tensor operations instead of launching
            # Python-level work for every token and bit.
            windows = route_codes.unfold(1, order, 1).permute(0, 1, 3, 2)
            address = torch.zeros((batch, windows.shape[1], routes), dtype=torch.long,
                                  device=route_codes.device)
            for exponent in range(order):
                address += windows[:, :, exponent, :].long() * (module.alphabet ** exponent)
            grad_out = grad_memory[:, order - 1:, order_index, :, :]
            table = module.tables[str(order)]
            idx_current = route_ids.view(1, 1, routes) * (module.alphabet ** order) + address
            current_value = table.weight[idx_current]
            for offset in range(order):
                current = windows[:, :, offset, :].long()
                direction = torch.where(
                    (current.unsqueeze(-1) & bit_values.view(1, 1, 1, -1)) == 0,
                    bit_values.view(1, 1, 1, -1),
                    -bit_values.view(1, 1, 1, -1),
                )
                changed = address.unsqueeze(2) + direction.permute(0, 1, 3, 2) * (module.alphabet ** offset)
                changed = changed.clamp(0, module.alphabet ** order - 1)
                idx_changed = route_ids.view(1, 1, 1, routes) * (module.alphabet ** order) + changed
                diff = table.weight[idx_changed] - current_value.unsqueeze(2)
                contribution = (grad_out.unsqueeze(2) * diff).sum(-1).permute(0, 1, 3, 2)
                positions = torch.arange(windows.shape[1], device=route_codes.device) + offset
                grad_logits.scatter_add_(
                    1,
                    positions.view(1, -1, 1, 1).expand(batch, -1, routes, module.bits),
                    contribution,
                )
        grad_logits = grad_logits * torch.sigmoid(route_logits) * (1.0 - torch.sigmoid(route_logits))
        return grad_memory, grad_logits, None, None


class LatentSymbolMemory(nn.Module):
    """Route-level exact n-gram memory following the public Lngram design."""

    def __init__(self, config, hidden_size: int, orders: Sequence[int]):
        super().__init__()
        self.hidden_size = hidden_size
        self.bits = int(getattr(config, "adapter_lngram_bits_per_route", 4))
        if self.bits < 1 or hidden_size % self.bits:
            raise ValueError("hidden_size must be divisible by adapter_lngram_bits_per_route")
        self.routes = hidden_size // self.bits
        self.alphabet = 1 << self.bits
        self.orders = tuple(int(order) for order in orders)
        # Reuse the common ADAPTER_HIDDEN_SIZE: for Lngram it is the
        # per-route value width, not the total model hidden size.
        # Lngram uses a compact per-latent-address value row.  The stage-5
        # setup uses ADAPTER_HIDDEN_SIZE=8; this fallback keeps direct model
        # construction aligned with that configuration.
        self.memory_dim = int(getattr(config, "adapter_hidden_size", None) or 8)
        self.route_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tables = nn.ModuleDict()
        for order in self.orders:
            # Unlike legacy Engram, Lngram has no arbitrary memory_size:
            # every route and every symbol n-gram has an exact table row.
            size = self.routes * self.alphabet ** order
            table = nn.Embedding(size, self.memory_dim)
            self.tables[str(order)] = table
        flat_dim = len(self.orders) * self.routes * self.memory_dim
        self.key_proj = nn.Linear(flat_dim, hidden_size, bias=False)
        self.value_proj = nn.Linear(flat_dim, hidden_size, bias=False)
        self.query_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        # Match Engram's optional causal depthwise residual path.  The
        # convolution operates on the already gated value stream, preserving
        # the [batch, sequence, hidden] shape and causal ordering.
        self.use_conv = bool(getattr(config, "adapter_use_conv", True))
        self.conv_kernel_size = int(getattr(config, "adapter_conv_kernel_size", 4))
        if self.conv_kernel_size < 1:
            raise ValueError("adapter_conv_kernel_size must be >= 1")
        self.conv_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.conv_kernel_size,
            groups=hidden_size,
            bias=False,
        )
        # Keep the newly enabled path an identity-free residual at start-up,
        # as in Engram; the direct value path remains available immediately.
        nn.init.zeros_(self.conv.weight)
        self.fixed_gate = getattr(config, "adapter_fixed_context_gate", None)
        self.alpha = (nn.Parameter(torch.tensor(float(getattr(config, "adapter_alpha_init", .1))))
                      if getattr(config, "adapter_use_alpha_layer", False) else None)

    def _causal_depthwise_conv(self, values: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(values).transpose(1, 2)
        conv_input = F.pad(conv_input, (self.conv_kernel_size - 1, 0))
        return F.silu(self.conv(conv_input)).transpose(1, 2)

    def route_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.route_proj(hidden).view(*hidden.shape[:2], self.routes, self.bits)

    def symbols_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        hard_bits = (logits >= 0).to(logits.dtype)
        weights = (2 ** torch.arange(self.bits, device=logits.device, dtype=logits.dtype))
        return (hard_bits * weights).sum(-1)

    def lookup(self, symbol_windows, valid_masks) -> torch.Tensor:
        pieces = []
        route_ids = torch.arange(self.routes, device=symbol_windows[0].device).view(1, 1, -1)
        for order, window, valid in zip(self.orders, symbol_windows, valid_masks):
            address = torch.zeros_like(window[:, :, 0, :], dtype=torch.long)
            stride = 1
            for offset in range(order):
                address += window[:, :, offset, :].long() * stride
                stride *= self.alphabet
            index = route_ids * (self.alphabet ** order) + address
            value = self.tables[str(order)](index)
            pieces.append(value * valid.unsqueeze(-1).unsqueeze(-1).to(value.dtype))
        return torch.stack(pieces, 2)

    def inject(self, hidden, memory):
        memory = memory.flatten(2)
        keys = self.key_proj(memory)
        if self.fixed_gate is None:
            gate = torch.sigmoid((self.query_norm(hidden) * self.key_norm(keys)).sum(-1, keepdim=True)
                                 / math.sqrt(self.hidden_size))
        else:
            gate = torch.full_like(hidden[..., :1], float(self.fixed_gate))
        values = gate * self.value_proj(memory)
        delta = values
        if self.use_conv:
            delta = delta + self._causal_depthwise_conv(values)
        return delta if self.alpha is None else self.alpha * delta


def causal_symbol_windows(symbols: torch.Tensor, orders: Sequence[int]):
    windows, valid = [], []
    batch, length, routes = symbols.shape
    for order in orders:
        padded = torch.nn.functional.pad(symbols, (0, 0, order - 1, 0))
        windows.append(padded.unfold(1, order, 1).permute(0, 1, 3, 2))
        positions = torch.arange(length, device=symbols.device).view(1, -1)
        valid.append((positions >= order - 1).expand(batch, -1))
    return windows, valid


class LngramMemoryAdapter(BaseMemoryAdapter):
    def __init__(self, config, hidden_size: int):
        super().__init__(hidden_size)
        max_order = int(getattr(config, "adapter_ngram_order", 3))
        self.memory = LatentSymbolMemory(config, hidden_size, range(2, max_order + 1))

    def forward(self, hidden_states, input_ids=None, attention_mask=None, layer_idx=None, **kwargs):
        route_hidden = kwargs.get("route_hidden_states", hidden_states)
        logits = self.memory.route_logits(route_hidden)
        symbols = self.memory.symbols_from_logits(logits)
        windows, valid = causal_symbol_windows(symbols, self.memory.orders)
        if attention_mask is not None and attention_mask.ndim == 2:
            valid = [mask & attention_mask[:, -hidden_states.shape[1]:].bool() for mask in valid]
        memory = self.memory.lookup(windows, valid)
        memory = _LatentLookupSurrogate.apply(memory, logits, symbols, self.memory)
        return self.memory.inject(hidden_states, memory)
