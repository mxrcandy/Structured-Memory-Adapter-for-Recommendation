from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseMemoryAdapter


class EngramMemoryAdapter(BaseMemoryAdapter):
    def __init__(
        self,
        hidden_size: int,
        adapter_hidden_size: int,
        memory_size: int,
        ngram_order: int = 2,
        topk: int = 4,
        hash_heads: Optional[int] = None,
        conv_kernel_size: int = 4,
        use_alpha_layer: bool = False,
        alpha_init: float = 0.1,
        fixed_context_gate: Optional[float] = None,
        use_conv: bool = True,
        use_sid_span: bool = False,
        sid_token_start: Optional[int] = None,
        sid_token_end: Optional[int] = None,
    ):
        super().__init__(hidden_size)
        self.memory_size = self._next_prime(memory_size)
        self.ngram_order = max(2, ngram_order)
        self.hash_heads = hash_heads if hash_heads is not None else topk
        self.hash_heads = max(1, self.hash_heads)
        self.head_dim = adapter_hidden_size
        self.orders = tuple(range(2, self.ngram_order + 1))
        self.num_tables = len(self.orders) * self.hash_heads
        self.memory_dim = self.num_tables * self.head_dim
        if fixed_context_gate is not None and not 0.0 <= fixed_context_gate <= 1.0:
            raise ValueError("fixed_context_gate must be in [0, 1]")
        if use_sid_span and (sid_token_start is None or sid_token_end is None):
            raise ValueError("SID span routing requires sid_token_start and sid_token_end")
        if sid_token_start is not None and sid_token_end is not None and sid_token_start > sid_token_end:
            raise ValueError("sid_token_start must not exceed sid_token_end")
        self.fixed_context_gate = fixed_context_gate
        self.use_conv = use_conv
        self.use_sid_span = use_sid_span
        self.sid_token_start = sid_token_start
        self.sid_token_end = sid_token_end
        self.alpha_layer = (
            nn.Parameter(torch.tensor([float(alpha_init)])) if use_alpha_layer else None
        )

        self.memories = nn.ModuleList(nn.Embedding(self.memory_size, self.head_dim) for _ in range(self.num_tables))
        self.key_proj = nn.Linear(self.memory_dim, hidden_size, bias=False)
        self.value_proj = nn.Linear(self.memory_dim, hidden_size, bias=False)
        self.query_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.conv_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.conv_kernel_size = conv_kernel_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=conv_kernel_size,
            groups=hidden_size,
            bias=False,
        )
        nn.init.zeros_(self.conv.weight)

        self.order_salts = tuple(
            0x9E3779B1 + 0x85EBCA77 * order for order in self.orders
        )
        self.head_salts = tuple(
            0xC2B2AE3D + 0x27D4EB2F * head for head in range(self.hash_heads)
        )

    @staticmethod
    def _next_prime(value: int) -> int:
        def is_prime(candidate: int) -> bool:
            if candidate < 2:
                return False
            if candidate == 2:
                return True
            if candidate % 2 == 0:
                return False
            divisor = 3
            while divisor * divisor <= candidate:
                if candidate % divisor == 0:
                    return False
                divisor += 2
            return True

        candidate = max(2, int(value))
        while not is_prime(candidate):
            candidate += 1
        return candidate

    def _hash_ngram(self, tokens: torch.LongTensor, order_idx: int, head_idx: int):
        salt = self.order_salts[order_idx] ^ self.head_salts[head_idx]
        hashed = torch.full(tokens.shape[:-1], salt, dtype=torch.long, device=tokens.device)
        for offset in range(tokens.shape[-1]):
            token = tokens[..., offset].long() + 1
            hashed = torch.bitwise_xor(
                hashed * 0x9E3779B1,
                token * (self.head_salts[head_idx] + offset + 1),
            )
        # Preserve the historical abs-based address mapping, while handling
        # the one int64 value for which abs() remains negative.
        min_int = torch.iinfo(hashed.dtype).min
        safe_abs = torch.where(hashed == min_int, torch.full_like(hashed, min_int + 1), hashed.abs())
        return torch.remainder(safe_abs, self.memory_size)

    def _make_lookup_ids(
        self,
        input_ids: Optional[torch.LongTensor],
        seq_len: int,
        device,
    ):
        if input_ids is None:
            positions = torch.arange(seq_len, device=device).view(1, seq_len, 1)
            return positions.expand(1, seq_len, self.num_tables) % self.memory_size

        recent_ids = input_ids[:, -seq_len:].long().to(device)
        lookup_ids = []
        for order_idx, order in enumerate(self.orders):
            padded = F.pad(recent_ids, (order - 1, 0), value=0)
            ngrams = padded.unfold(dimension=1, size=order, step=1)
            for head_idx in range(self.hash_heads):
                lookup_ids.append(self._hash_ngram(ngrams, order_idx, head_idx))
        return torch.stack(lookup_ids, dim=-1)

    def _lookup_memory(self, lookup_ids: torch.LongTensor) -> torch.Tensor:
        pieces = [
            memory(lookup_ids[..., table_idx])
            for table_idx, memory in enumerate(self.memories)
        ]
        return torch.cat(pieces, dim=-1)

    def _causal_depthwise_conv(self, values: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(values).transpose(1, 2)
        conv_input = F.pad(conv_input, (self.conv_kernel_size - 1, 0))
        return F.silu(self.conv(conv_input)).transpose(1, 2)

    def _context_gate(self, hidden_states: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if self.fixed_context_gate is not None:
            return torch.full_like(keys[..., :1], self.fixed_context_gate)
        query = self.query_norm(hidden_states)
        key = self.key_norm(keys)
        return torch.sigmoid(
            (query * key).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        )

    def _apply_output_controls(
        self,
        delta: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        if self.use_sid_span:
            if input_ids is None:
                return torch.zeros_like(delta)
            recent_ids = input_ids[:, -delta.shape[1]:].to(delta.device)
            sid_mask = (
                (recent_ids >= self.sid_token_start)
                & (recent_ids <= self.sid_token_end)
            ).unsqueeze(-1)
            delta = delta * sid_mask.to(dtype=delta.dtype)
        if self.alpha_layer is not None:
            delta = self.alpha_layer * delta
        return delta

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        lookup_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        _, seq_len, _ = hidden_states.shape
        if lookup_ids is None:
            lookup_ids = self._make_lookup_ids(input_ids, seq_len, hidden_states.device)
        else:
            if lookup_ids.shape[:2] != hidden_states.shape[:2]:
                raise ValueError("External Engram lookup ids must match query shape")
            lookup_ids = lookup_ids.to(hidden_states.device)
        memory = self._lookup_memory(lookup_ids)
        keys = self.key_proj(memory)
        values = self.value_proj(memory)

        context_gate = self._context_gate(hidden_states, keys)
        values = context_gate * values
        delta = values
        if self.use_conv:
            delta = delta + self._causal_depthwise_conv(values)
        return self._apply_output_controls(delta, input_ids)
