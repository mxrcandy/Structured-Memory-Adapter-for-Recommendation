"""Globally sample-weighted next-a loss for data-parallel/FSDP training."""
import math

import torch
import torch.distributed as dist


def configure_itm_aux_training(config, weight=None):
    """An omitted CLI value inherits saved training settings; zero keeps heads."""
    if weight is None:
        weight = getattr(config, 'adapter_itm_aux_loss_weight', 0.)
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('adapter_itm_aux_loss_weight must be finite and nonnegative')
    head = bool(getattr(config, 'adapter_itm_aux_head', False)) or weight > 0
    if head and (getattr(config, 'adapter_type', None) != 'engram_structure'
                 or not getattr(config, 'adapter_use_itm', True)
                 or getattr(config, 'adapter_itm_mode', 'default') != 'exact'):
        raise ValueError('next-a auxiliary head requires engram_structure with exact ITM enabled')
    config.adapter_itm_aux_loss_weight = weight
    config.adapter_itm_aux_head = head


def normalize_itm_aux(stats):
    """Return global-mean CE with correct gradients under averaged DDP/FSDP.

    stats is [sum CE, sum correct, count], already averaged across ITM layers.
    Detached all-reduce is intentional: local differentiable sums are scaled by
    world_size / global_count to compensate the gradient collective's averaging.
    The scalar's displayed value is the same global mean on every rank.
    """
    totals = stats.detach().float().clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denominator = totals[2].clamp_min(1)
    local_loss = stats[0] * (world / denominator)
    mean = totals[0] / denominator
    loss = local_loss + (mean - local_loss.detach())
    values = totals.cpu().tolist()
    return loss, dict(itm_aux_loss=values[0] / max(values[2], 1),
                      itm_aux_top1=values[1] / max(values[2], 1),
                      itm_aux_count=values[2])
