import math
from typing import Dict, Optional, Set

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_hidden = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return base_out + lora_out * self.lora_scaling


def _parse_csv(value: Optional[str]) -> Optional[Set[str]]:
    if value in (None, "", "all"):
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _parse_int_csv(value: Optional[str]) -> Optional[Set[int]]:
    parsed = _parse_csv(value)
    if parsed is None:
        return None
    return {int(item) for item in parsed}


def _layer_index_from_name(name: str) -> Optional[int]:
    parts = name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except (ValueError, IndexError):
                return None
    return None


def apply_lora_to_model(
    model: nn.Module,
    target_modules: str,
    layers: Optional[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    targets = _parse_csv(target_modules) or set()
    selected_layers = _parse_int_csv(layers)
    replacements: Dict[str, LoRALinear] = {}

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in targets:
            continue
        layer_idx = _layer_index_from_name(module_name)
        if selected_layers is not None and layer_idx not in selected_layers:
            continue
        replacements[module_name] = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

    for module_name, replacement in replacements.items():
        parent = model
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
    return len(replacements)


def merge_lora_state_dict(state_dict: Dict[str, torch.Tensor], lora_alpha: float, lora_rank: int) -> Dict[str, torch.Tensor]:
    merged = dict(state_dict)
    scaling = float(lora_alpha) / float(lora_rank)
    lora_a_keys = [key for key in list(merged) if key.endswith(".lora_A")]

    for a_key in lora_a_keys:
        prefix = a_key[: -len(".lora_A")]
        b_key = f"{prefix}.lora_B"
        w_key = f"{prefix}.weight"
        if b_key not in merged or w_key not in merged:
            continue
        weight = merged[w_key]
        delta = torch.matmul(merged[b_key].to(torch.float32), merged[a_key].to(torch.float32)) * scaling
        merged[w_key] = (weight.to(torch.float32) + delta).to(weight.dtype)
        del merged[a_key]
        del merged[b_key]

    return merged
