"""Distributed training utilities for FSDP model sharding and checkpoint loading."""

from typing import Any, Dict, Optional
from pathlib import Path
import re

import torch
import torch.distributed as dist
from torch import nn
from onerec_llm.utils.ds_utils import format_dict_or_list
from onerec_llm.utils.distributed import get_world_size_and_rank

from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import distribute_tensor
from torch.distributed.device_mesh import DeviceMesh


def initialize_allowed_missing_parameter(
    name: str,
    shape,
    memory_adapter_alpha_init: float = 0.1,
    memory_init: str = "default",
    memory_init_std: float = 0.01,
    model=None,
    prototype_cache=None,
) -> torch.Tensor:
    """Initialize trainable params that are absent from the base checkpoint."""
    tensor = torch.empty(shape)
    is_memory_table = (
        ".memory_adapter." in name
        and name.endswith(".weight")
        and (".memories." in name or ".tables." in name)
    )
    if is_memory_table and memory_init == "prototype":
        prototype_value = _prototype_missing_tensor(name, shape, model, prototype_cache)
        if prototype_value is not None:
            return prototype_value
    if is_memory_table and memory_init == "zero":
        return torch.zeros(shape)
    if is_memory_table and memory_init == "normal":
        return torch.randn(shape) * float(memory_init_std)
    if is_memory_table and memory_init == "random":
        return torch.empty(shape).uniform_(-0.02, 0.02)
    tensor.uniform_(0.0, 0.1)
    if name.endswith(".lora_B"):
        nn.init.zeros_(tensor)
    elif ".houlsby_adapter." in name and ".up." in name:
        nn.init.zeros_(tensor)
    elif ".memory_adapter." in name and name.endswith(".alpha_layer"):
        nn.init.constant_(tensor, memory_adapter_alpha_init)
    elif ".memory_adapter." in name and name.endswith("_norm.weight"):
        nn.init.ones_(tensor)
    elif tensor.ndim >= 2:
        nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu')
    else:
        nn.init.zeros_(tensor)
    return tensor


def _prototype_missing_tensor(name, shape, model, cache):
    """Return a prototype row table for a missing adapter parameter, if applicable."""
    config = getattr(model, "config", None)
    path = getattr(config, "adapter_memory_prototype_path", None)
    if not path or getattr(config, "adapter_memory_init", "default") != "prototype":
        return None
    match = re.search(r"\.memory_adapter\.(?:(spm|itm)\.)?(?:memory\.)?memories\.(\d+)\.weight$", name)
    table_match = re.search(r"\.memory_adapter\.(?:(spm|itm)\.)?(?:memory\.)?tables\.(\d+)\.weight$", name)
    if not match and not table_match:
        return None
    if table_match:
        branch = table_match.group(1)
        order = int(table_match.group(2))
        if cache is None:
            cache = {}
        if "payload" not in cache:
            cache["payload"] = torch.load(Path(path), map_location="cpu", weights_only=False)
        payload = cache["payload"]
        tables = payload.get("tables", {})
        if branch is None:
            table = tables.get(str(order), tables.get(order))
        else:
            branch_tables = tables.get(branch, {})
            table = branch_tables.get(str(order), branch_tables.get(order))
        if table is None:
            return None
        table = torch.as_tensor(table)
        if tuple(table.shape) != tuple(shape):
            raise ValueError(f"Prototype shape mismatch for {name}: file {tuple(table.shape)}, model {tuple(shape)}")
        return table
    branch, index = match.group(1), int(match.group(2))
    if cache is None:
        cache = {}
    if "payload" not in cache:
        cache["payload"] = torch.load(Path(path), map_location="cpu", weights_only=False)
    payload = cache["payload"]
    tables = payload.get("tables", {})
    if branch is None:
        ordered = []
        for order in range(2, int(getattr(config, "adapter_ngram_order", 3)) + 1):
            ordered.extend(torch.as_tensor(tables[str(order)])[h]
                           for h in range(int(getattr(config, "adapter_topk", 8))))
    else:
        branch_tables = tables.get(branch, {})
        start_order = 2 if branch == "spm" else 1
        end_order = int(getattr(config, "adapter_ngram_order", 3)) if branch == "spm" else int(getattr(config, "adapter_itm_order", 1))
        ordered = []
        for order in range(start_order, end_order + 1):
            ordered.extend(torch.as_tensor(branch_tables[str(order)])[h]
                           for h in range(int(getattr(config, "adapter_topk", 4))))
    if index >= len(ordered) or tuple(ordered[index].shape) != tuple(shape):
        raise ValueError(f"Prototype shape/index mismatch for {name}: expected {tuple(shape)}")
    return ordered[index]

def shard_model(
    model: nn.Module,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
    fp32_weight: bool = True,
    model_class: str = 'Qwen3ForCausalLM',
    fp32_reduce: bool = True
) -> None:
    """Shard a model with FSDP using the PyTorch Distributed fully_shard API.
    
    Args:
        model: Model to shard with FSDP.
        cpu_offload: If True, FSDP will offload parameters to CPU.
        reshard_after_forward: Whether to reshard after forward pass.
        dp_mesh: Device mesh for FSDP sharding under multiple parallelism.
        fp32_weight: If True, use fp32 for weights with bfloat16 params.
        model_class: Model class name. Currently only supports 'Qwen3ForCausalLM'.
        fp32_reduce: If True, use fp32 for gradient reduction.
    """
    fsdp_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    
    if fp32_weight:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32 if fp32_reduce else torch.bfloat16
        )
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if model_class == 'Qwen3ForCausalLM':
        layers = list(model.model.layers)
    else:
        raise ValueError(f"Unsupported model_class: {model_class}")
    
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    
    fully_shard(model, **fsdp_kwargs)

    # Set up forward prefetch for layers
    prev = None
    for layer in reversed(layers):
        if prev is not None:
            layer.set_modules_to_forward_prefetch([prev])
        prev = layer
    model.set_modules_to_forward_prefetch([prev])


def load_from_full_model_state_dict(
    model: "FSDPModule",
    full_sd: Dict[str, Any],
    allow_random_init_params: Optional[str] = None,
    use_tie_weights: bool = False,
    memory_adapter_alpha_init: float = 0.1,
    memory_init: str = "default",
    memory_init_std: float = 0.01,
    require_itm_aux_head: bool = False,
    require_itm_prior: bool = False,
) -> None:
    """Load full state dict into an FSDP-sharded model.
    
    Args:
        model: FSDP-sharded model to load into.
        full_sd: Full (unsharded) state dictionary.
        allow_random_init_params: Comma-separated parameter names to randomly initialize
            if not found in full_sd. Default: None.
        use_tie_weights: If True, tie lm_head.weight to model.embed_tokens.weight.
    """
    if isinstance(allow_random_init_params, str):
        allow_random_init_params = allow_random_init_params.split(',')
    allow_random_init_params = allow_random_init_params or []
    
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    
    if dist.get_rank() == 0:
        if use_tie_weights:
            full_sd['lm_head.weight'] = full_sd['model.embed_tokens.weight']

        extra_meta_sharded_sd = set(meta_sharded_sd.keys()) - set(full_sd.keys())
        extra_full_ds = set(full_sd.keys()) - set(meta_sharded_sd.keys())
        
        extra_meta_sharded_sd = {
            k: (v.shape, v.device, v.dtype) 
            for k, v in meta_sharded_sd.items() 
            if k in extra_meta_sharded_sd
        }
        extra_full_ds = {
            k: (v.shape, v.device, v.dtype) 
            for k, v in full_sd.items() 
            if k in extra_full_ds
        }

        if require_itm_prior:
            missing_prior = [k for k in extra_meta_sharded_sd if '.memory_adapter.itm.' in k]
            extra_prior = [k for k in extra_full_ds if '.memory_adapter.itm.' in k]
            if missing_prior or extra_prior:
                raise ValueError(f'Saved successor prior weights mismatch: missing={missing_prior}, extra={extra_prior}')
        if require_itm_aux_head:
            missing_heads = [k for k in extra_meta_sharded_sd if '.itm.next_a_head.' in k]
            if missing_heads:
                raise ValueError(f'Checkpoint is missing saved next-a head weights: {missing_heads}')

        device0 = full_sd[list(full_sd)[0]]
        allowed_missing = {}
        prototype_cache = {}
        for k in extra_meta_sharded_sd:
            is_allowed = (
                k in allow_random_init_params
                or k.startswith("model.layers.") and ".memory_adapter." in k
                or k.startswith("model.layers.") and (k.endswith(".lora_A") or k.endswith(".lora_B"))
                or k.startswith("model.layers.") and ".houlsby_adapter." in k
            )
            if is_allowed:
                full_sd[k] = initialize_allowed_missing_parameter(
                    k, extra_meta_sharded_sd[k][0],
                    memory_adapter_alpha_init=memory_adapter_alpha_init,
                    memory_init=memory_init,
                    memory_init_std=memory_init_std,
                    model=model,
                    prototype_cache=prototype_cache)
                full_sd[k] = full_sd[k].to(device0)
                allowed_missing[k] = extra_meta_sharded_sd[k]
        for k in allowed_missing:
            extra_meta_sharded_sd.pop(k, None)

        assert len(extra_meta_sharded_sd) == 0 and len(extra_full_ds) == 0, (
            f"Sharded State Dict doesn't equal to Full State Dict, "
            f"{len(meta_sharded_sd)} vs {len(full_sd)}\n"
            f"extra_meta_sharded_sd={format_dict_or_list(extra_meta_sharded_sd)}, "
            f"extra_full_ds={format_dict_or_list(extra_full_ds)}"
        )
        assert sorted(list(meta_sharded_sd.keys())) == sorted(list(full_sd.keys())), \
            "Keys of Sharded State Dict doesn't equal to Full State Dict"

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        if dist.get_rank() == 0:
            full_tensor = full_sd[param_name].detach().cuda().type(sharded_meta_param.dtype)
        else:
            full_tensor = torch.empty(
                sharded_meta_param.size(),
                device="cuda",
                dtype=sharded_meta_param.dtype,
            )
        
        mesh = sharded_meta_param.device_mesh
        dist.broadcast(full_tensor, src=0, group=mesh.get_group(0))
        dist.barrier()
        
        sharded_tensor = distribute_tensor(
            full_tensor, mesh, sharded_meta_param.placements
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)

    model.load_state_dict(sharded_sd, assign=True)
