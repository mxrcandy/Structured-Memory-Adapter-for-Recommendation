from __future__ import annotations

from typing import Optional

import torch


def _print_rank_0(message: str) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message)
    else:
        print(message)


def _parse_csv_set(value: Optional[str]) -> Optional[set[str]]:
    if value in (None, "", "all"):
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _layer_index_from_param_name(name: str) -> Optional[int]:
    parts = name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except (IndexError, ValueError):
                return None
    return None


def _layer_count_from_model(model: torch.nn.Module) -> int:
    layer_indices = {
        layer_idx
        for name, _ in model.named_parameters()
        for layer_idx in [_layer_index_from_param_name(name)]
        if layer_idx is not None
    }
    if not layer_indices:
        return 0
    return max(layer_indices) + 1


def _parse_layer_selector(value: Optional[str], num_layers: int) -> set[int]:
    if value in (None, ""):
        raise ValueError("adapter_partial requires --partial_train_layers; use last:N, comma-separated ids, or all")
    if value == "all":
        return set(range(num_layers))
    value = str(value).strip()
    if value.startswith("last:"):
        count = int(value.split(":", 1)[1])
        if count <= 0:
            raise ValueError(f"last:N selector requires N > 0, got {value}")
        if num_layers <= 0:
            raise ValueError("Cannot use last:N partial layer selector when model has no decoder layers")
        return set(range(max(0, num_layers - count), num_layers))
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _module_leaf_from_param_name(name: str) -> Optional[str]:
    parts = name.split(".")
    if len(parts) < 2:
        return None
    return parts[-2]


def _is_partial_trainable_param(name: str, selected_layers: set[int], target_modules: set[str]) -> bool:
    layer_idx = _layer_index_from_param_name(name)
    if layer_idx not in selected_layers:
        return False
    module_leaf = _module_leaf_from_param_name(name)
    return module_leaf in target_modules


def configure_trainable_parameters(model: torch.nn.Module, args) -> None:
    if getattr(args, "finetune_mode", "full") == "sid_only":
        if args.start_optimize_embedding_index <= 0:
            raise ValueError("sid_only requires --start_optimize_embedding_index > 0")
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
        _print_rank_0(
            "SID-only mode will optimize tied itemic embedding rows "
            f"from index {args.start_optimize_embedding_index}; older token "
            "rows are restored after each optimizer step by EmbeddingGradientMasker."
        )
    elif getattr(args, "finetune_mode", "full") == "adapter":
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            # Keep the frozen decoder frozen.  The previous substring check
            # also matched every decoder ``mlp.gate_proj`` parameter.
            if "memory_adapter" in name:
                param.requires_grad = True
            elif (
                args.start_optimize_embedding_index > 0
                and ("embed_tokens" in name or "lm_head" in name)
            ):
                param.requires_grad = True
        if args.start_optimize_embedding_index > 0:
            _print_rank_0(
                "Adapter mode will also optimize tied itemic embedding rows "
                f"from index {args.start_optimize_embedding_index}; older token "
                "rows are restored after each optimizer step by EmbeddingGradientMasker."
            )
    elif getattr(args, "finetune_mode", "full") == "lora":
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                param.requires_grad = True
            elif (
                args.start_optimize_embedding_index > 0
                and ("embed_tokens" in name or "lm_head" in name)
            ):
                param.requires_grad = True
        if args.start_optimize_embedding_index > 0:
            _print_rank_0(
                "LoRA mode will also optimize tied itemic embedding rows "
                f"from index {args.start_optimize_embedding_index}; older token "
                "rows are restored after each optimizer step by EmbeddingGradientMasker."
            )
    elif getattr(args, "finetune_mode", "full") == "adapter_lora":
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "memory_adapter" in name:
                param.requires_grad = True
            elif name.endswith(".lora_A") or name.endswith(".lora_B"):
                param.requires_grad = True
            elif (
                args.start_optimize_embedding_index > 0
                and ("embed_tokens" in name or "lm_head" in name)
            ):
                param.requires_grad = True
        if args.start_optimize_embedding_index > 0:
            _print_rank_0(
                "Adapter+LoRA mode will also optimize tied itemic embedding rows "
                f"from index {args.start_optimize_embedding_index}; older token "
                "rows are restored after each optimizer step by EmbeddingGradientMasker."
            )
    elif getattr(args, "finetune_mode", "full") == "houlsby_adapter":
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "houlsby_adapter" in name:
                param.requires_grad = True
            elif (
                args.start_optimize_embedding_index > 0
                and ("embed_tokens" in name or "lm_head" in name)
            ):
                param.requires_grad = True
        if args.start_optimize_embedding_index > 0:
            _print_rank_0(
                "Houlsby Adapter mode will also optimize tied itemic embedding rows "
                f"from index {args.start_optimize_embedding_index}; older token "
                "rows are restored after each optimizer step by EmbeddingGradientMasker."
            )
    elif getattr(args, "finetune_mode", "full") == "adapter_partial":
        target_modules = _parse_csv_set(getattr(args, "partial_train_target_modules", None)) or set()
        selected_layers = _parse_layer_selector(
            getattr(args, "partial_train_layers", None),
            _layer_count_from_model(model),
        )
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "memory_adapter" in name:
                param.requires_grad = True
            elif _is_partial_trainable_param(name, selected_layers, target_modules):
                param.requires_grad = True
            elif (
                args.start_optimize_embedding_index > 0
                and ("embed_tokens" in name or "lm_head" in name)
            ):
                param.requires_grad = True
        _print_rank_0(
            "Adapter partial mode will unfreeze decoder layers "
            f"{sorted(selected_layers)} for target modules {sorted(target_modules)}."
        )
        if args.start_optimize_embedding_index > 0:
            _print_rank_0(
                "Adapter partial mode will also optimize tied itemic embedding rows "
                f"from index {args.start_optimize_embedding_index}; older token "
                "rows are restored after each optimizer step by EmbeddingGradientMasker."
            )
    elif getattr(args, "finetune_mode", "full") == "adapter_full":
        for _, param in model.named_parameters():
            param.requires_grad = True
        _print_rank_0("Adapter full mode will optimize all backbone and memory adapter parameters.")
    elif getattr(args, "freeze_llm", False):
        assert args.start_optimize_embedding_index > 0
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            _print_rank_0(f"Trainable parameter: {name}")
    _print_rank_0("=" * 50)
