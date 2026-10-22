"""Checkpoint to HuggingFace Format Converter

This module provides utilities to convert PyTorch checkpoints (DCP or .pth files)
to HuggingFace format (safetensors or bin files with sharding support).
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Union

PRETRAIN_ROOT = Path(__file__).resolve().parents[2]
if str(PRETRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_ROOT))

import torch
import tqdm
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from onerec_llm.models.lora import merge_lora_state_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
SHARD_FNAME_TEMPLATE = "model-{cpt_idx}-of-{num_shards}"
BYTES_PER_GB = 1024 * 1024 * 1024
DEFAULT_MAX_GB_PER_SHARD = 5
DEFAULT_DTYPE = "bf16"
REPO_ROOT = Path(__file__).resolve().parents[3]

# Common HuggingFace config files to copy
HF_CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",  # SentencePiece tokenizer model file
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "preprocessor_config.json",  # For vision models
]


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype.

    Args:
        dtype_str: Data type string ("fp32", "fp16", "bf16")

    Returns:
        Corresponding torch.dtype

    Raises:
        ValueError: If dtype_str is not supported
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def _extract_state_dict_from_checkpoint(checkpoint: Dict, model_only: bool = True) -> Dict[str, torch.Tensor]:
    """Extract state_dict from checkpoint with various structures.

    Args:
        checkpoint: Checkpoint dictionary
        model_only: Whether to extract only model weights

    Returns:
        State dictionary containing model weights
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    # Check for nested DCP-like structure
    if model_only and "app" in checkpoint and "model" in checkpoint["app"]:
        logger.info("Found nested structure: checkpoint['app']['model']")
        return checkpoint["app"]["model"]
    elif "model" in checkpoint:
        logger.info("Found structure: checkpoint['model']")
        return checkpoint["model"]
    elif "state_dict" in checkpoint:
        logger.info("Found structure: checkpoint['state_dict']")
        return checkpoint["state_dict"]
    else:
        # Assume entire dict is the state_dict
        logger.info("Using entire checkpoint as state_dict")
        return checkpoint


def _convert_state_dict_to_shards(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Union[str, os.PathLike],
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE
) -> None:
    """Convert state_dict to sharded safetensors or bin files.

    Args:
        state_dict: State dictionary containing model weights
        output_dir: Output directory for sharded files
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion ("fp32", "fp16", "bf16", default: "bf16")

    Raises:
        ValueError: If dtype is not supported
    """
    torch_dtype = _get_torch_dtype(dtype)
    logger.info(f"Converting state_dict to {dtype} format")

    # Convert data types
    logger.info("Converting tensor data types...")
    for key in tqdm.tqdm(state_dict.keys(), desc="Converting dtypes"):
        state_dict[key] = state_dict[key].to(torch_dtype)

    # Split into shards
    logger.info(f"Splitting state_dict into shards (max {max_gb_per_shard} GB per shard)...")
    split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
    shard_idx = 0
    total_size = 0
    current_size = 0

    max_bytes_per_shard = max_gb_per_shard * BYTES_PER_GB

    for key, weight in tqdm.tqdm(state_dict.items(), desc="Creating shards"):
        if shard_idx not in split_state_dicts:
            split_state_dicts[shard_idx] = {}

        split_state_dicts[shard_idx][key] = weight
        weight_size = weight.numel() * weight.element_size()
        current_size += weight_size
        total_size += weight_size

        if current_size >= max_bytes_per_shard:
            shard_idx += 1
            current_size = 0

    # Write shard files
    num_shards = len(split_state_dicts)
    weight_map: Dict[str, str] = {}
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {num_shards} shard files...")
    for shard_idx, shard_state_dict in tqdm.tqdm(split_state_dicts.items(), desc="Writing shards"):
        shard_name = SHARD_FNAME_TEMPLATE.format(
            cpt_idx=f"{shard_idx}".zfill(5),
            num_shards=f"{num_shards}".zfill(5)
        )

        if use_safetensor:
            shard_path = output_path_obj / f"{shard_name}.safetensors"
            save_file(shard_state_dict, shard_path, metadata={"format": "pt"})
        else:
            shard_path = output_path_obj / f"{shard_name}.bin"
            torch.save(shard_state_dict, shard_path)

        # Update weight map
        shard_filename = shard_path.name
        for key in shard_state_dict.keys():
            weight_map[key] = shard_filename

        shard_size_gb = os.path.getsize(shard_path) / BYTES_PER_GB
        logger.info(f"Shard {shard_idx + 1}/{num_shards}: {shard_size_gb:.2f} GiB saved to {shard_path}")

    # Write index file
    index_filename = "model.safetensors.index.json" if use_safetensor else "model.bin.index.json"
    index_path = output_path_obj / index_filename

    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Index file saved to {index_path}")
    logger.info(f"Total model size: {total_size / BYTES_PER_GB:.2f} GiB")


def pth_to_hf_format(
    pth_file_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE,
    merge_lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[float] = None,
) -> None:
    """Convert .pth file to HuggingFace format (safetensors or bin files).

    Args:
        pth_file_path: Path to .pth checkpoint file
        output_dir: Output directory for converted files
        model_only: Whether to extract only model weights (default: True)
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion (default: "bf16")

    Raises:
        FileNotFoundError: If pth_file_path does not exist
        ValueError: If pth_file_path is not a .pth file or has unsupported format

    .. warning::
        To avoid OOM, it's recommended to run this function on a single rank/process.
    """
    pth_path = Path(pth_file_path)

    if not pth_path.exists():
        raise FileNotFoundError(f"PTH file not found: {pth_path}")

    if pth_path.suffix != ".pth":
        raise ValueError(f"Expected .pth file, got: {pth_path.suffix}")

    logger.info(f"Loading PTH file from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location="cpu")

    # Extract state_dict from checkpoint
    state_dict = _extract_state_dict_from_checkpoint(checkpoint, model_only=model_only)
    logger.info(f"Loaded state_dict with {len(state_dict)} keys")
    if merge_lora:
        if lora_rank is None or lora_alpha is None:
            raise ValueError("--merge_lora requires --lora_rank and --lora_alpha")
        state_dict = merge_lora_state_dict(state_dict, lora_alpha=lora_alpha, lora_rank=lora_rank)
        logger.info("Merged LoRA weights into base linear weights")

    # Convert to HuggingFace format
    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype
    )


def dcp_to_hf_format(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE,
    merge_lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[float] = None,
) -> None:
    """Convert DCP (Distributed Checkpoint) to HuggingFace format.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint
        output_dir: Output directory for converted files
        model_only: Whether to extract only model weights (default: True)
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion (default: "bf16")

    Raises:
        FileNotFoundError: If dcp_checkpoint_dir does not exist

    .. warning::
        To avoid OOM, it's recommended to run this function on a single rank/process.
    """
    dcp_path = Path(dcp_checkpoint_dir)

    if not dcp_path.exists():
        raise FileNotFoundError(f"DCP checkpoint directory not found: {dcp_path}")

    if not dcp_path.is_dir():
        raise ValueError(f"Expected directory, got: {dcp_path}")

    logger.info(f"Loading DCP checkpoint from {dcp_path}...")
    state_dict: STATE_DICT_TYPE = {}

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(str(dcp_path)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    logger.info("DCP checkpoint loaded successfully")

    if model_only:
        if "app" not in state_dict or "model" not in state_dict["app"]:
            raise ValueError("Expected 'app.model' in DCP checkpoint when model_only=True")
        state_dict = state_dict["app"]["model"]
        logger.info(f"Extracted model state_dict with {len(state_dict)} keys")
    if merge_lora:
        if lora_rank is None or lora_alpha is None:
            raise ValueError("--merge_lora requires --lora_rank and --lora_alpha")
        state_dict = merge_lora_state_dict(state_dict, lora_alpha=lora_alpha, lora_rank=lora_rank)
        logger.info("Merged LoRA weights into base linear weights")

    # Convert to HuggingFace format
    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype
    )


def copy_hf_config_files(
    source_hf_model_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike]
) -> None:
    """Copy HuggingFace configuration files from source to output directory.

    Args:
        source_hf_model_path: Path to source HuggingFace model directory
        output_dir: Output directory where config files will be copied
    """
    source_path = Path(source_hf_model_path)
    output_path = Path(output_dir)

    if not source_path.exists():
        logger.warning(f"Source HuggingFace model path does not exist: {source_path}")
        return

    if not source_path.is_dir():
        logger.warning(f"Source path is not a directory: {source_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    copied_files = []

    # Copy known config files
    for config_file in HF_CONFIG_FILES:
        source_file = source_path / config_file
        if source_file.exists():
            dest_file = output_path / config_file
            if source_file.resolve() == dest_file.resolve():
                copied_files.append(config_file)
                logger.debug(f"Skipped copying {config_file}; source and destination are the same")
                continue
            shutil.copy2(source_file, dest_file)
            copied_files.append(config_file)
            logger.debug(f"Copied {config_file} to {output_path}")

    # Copy additional JSON and TXT files (may be config files)
    for pattern in ["*.json", "*.txt"]:
        for source_file in source_path.glob(pattern):
            # Skip already copied files and weight files
            if (source_file.name in copied_files or
                source_file.name.startswith("model-") or
                source_file.suffix in [".bin", ".safetensors"]):
                continue

            dest_file = output_path / source_file.name
            if source_file.resolve() == dest_file.resolve():
                continue
            if not dest_file.exists():  # Avoid overwriting already copied files
                shutil.copy2(source_file, dest_file)
                if source_file.name not in HF_CONFIG_FILES:
                    logger.debug(f"Copied additional file: {source_file.name}")

    if copied_files:
        logger.info(f"Successfully copied {len(copied_files)} config files from {source_path} to {output_path}")
    else:
        logger.warning(f"No config files found in {source_path}")


def _rewrite_remote_code_text(source_text: str, relative_dest: str) -> str:
    if relative_dest == "modeling_qwen3.py":
        source_text = source_text.replace(
            "from ..memory_adapter import build_memory_adapter",
            "from .memory_adapter import build_memory_adapter",
        )
        source_text = source_text.replace(
            "from ..memory_adapter.structure import StructuredDynamicCache, prepare_structure",
            "from .memory_adapter import StructuredDynamicCache, prepare_structure",
        )
        source_text = source_text.replace(
            "from ..houlsby_adapter import HoulsbyAdapter",
            "from .houlsby_adapter import HoulsbyAdapter",
        )
    return source_text


def _build_remote_memory_adapter_module() -> str:
    source_dir = REPO_ROOT / "pretrain/onerec_llm/models/memory_adapter"
    base_text = (source_dir / "base.py").read_text(encoding="utf-8")
    engram_text = (source_dir / "engram.py").read_text(encoding="utf-8")
    structure_text = (source_dir / "structure.py").read_text(encoding="utf-8")
    lngram_text = (source_dir / "lngram.py").read_text(encoding="utf-8")
    lngram_structure_text = (source_dir / "lngram_structure.py").read_text(encoding="utf-8")

    def strip_common_imports(text: str) -> str:
        lines = []
        for line in text.splitlines():
            if line == "from __future__ import annotations":
                continue
            if line == "from typing import Optional":
                continue
            if line == "from typing import Optional, Sequence":
                continue
            if line == "import torch":
                continue
            if line == "from torch import nn":
                continue
            if line == "from .base import BaseMemoryAdapter":
                continue
            if line.startswith("from .lngram import "):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    base_text = strip_common_imports(base_text)
    base_text = base_text.replace(
        '    if adapter_type == "engram":\n'
        "        from .engram import EngramMemoryAdapter\n\n"
        "        return EngramMemoryAdapter(",
        '    if adapter_type == "engram":\n'
        "        return EngramMemoryAdapter(",
    )

    base_text = base_text.replace("        from .structure import StructuredEngramMemoryAdapter\n", "")
    base_text = base_text.replace("        from .lngram_structure import StructuredLngramMemoryAdapter\n", "")
    base_text = base_text.replace("        from .lngram import LngramMemoryAdapter\n", "")

    build_fn_start = base_text.index("\ndef build_memory_adapter")
    base_classes_text = base_text[:build_fn_start].strip()
    build_fn_text = base_text[build_fn_start:].strip()

    module_parts = [
        "from __future__ import annotations",
        "",
        "from typing import Optional",
        "",
        "import torch",
        "from torch import nn",
        "",
        base_classes_text,
        "",
        strip_common_imports(engram_text),
        "",
        strip_common_imports(structure_text),
        "",
        strip_common_imports(lngram_text),
        "",
        strip_common_imports(lngram_structure_text),
        "",
        build_fn_text,
        "",
    ]
    return "\n".join(module_parts)


def copy_remote_code_bundle(output_dir: Union[str, os.PathLike]) -> None:
    """Copy the local Qwen3 remote-code bundle into a converted HF directory."""
    output_path = Path(output_dir)
    remote_code_files = [
        (REPO_ROOT / "pretrain/onerec_llm/models/qwen3/configuration_qwen3.py", "configuration_qwen3.py"),
        (REPO_ROOT / "pretrain/onerec_llm/models/qwen3/modeling_qwen3.py", "modeling_qwen3.py"),
        (REPO_ROOT / "pretrain/onerec_llm/models/houlsby_adapter.py", "houlsby_adapter.py"),
    ]

    for source_file, relative_dest in remote_code_files:
        if not source_file.exists():
            logger.warning(f"Remote code source file not found: {source_file}")
            continue
        dest_file = output_path / relative_dest
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(
            _rewrite_remote_code_text(source_file.read_text(encoding="utf-8"), relative_dest),
            encoding="utf-8",
        )
        logger.info(f"Copied remote code file: {source_file} -> {dest_file}")

    memory_adapter_file = output_path / "memory_adapter.py"
    memory_adapter_file.write_text(_build_remote_memory_adapter_module(), encoding="utf-8")
    logger.info(f"Wrote remote code file: {memory_adapter_file}")


def update_remote_code_config(output_dir: Union[str, os.PathLike]) -> None:
    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        logger.warning(f"Cannot update remote code config because config.json is missing: {config_path}")
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["auto_map"] = {
        "AutoConfig": "configuration_qwen3.Qwen3Config",
        "AutoModel": "modeling_qwen3.Qwen3Model",
        "AutoModelForCausalLM": "modeling_qwen3.Qwen3ForCausalLM",
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Updated remote code config in {config_path}")


def update_adapter_config(
    output_dir: Union[str, os.PathLike],
    adapter_type: Optional[str],
    adapter_layers: Optional[str],
    adapter_hidden_size: Optional[int],
    adapter_spm_hidden_size: Optional[int],
    adapter_itm_hidden_size: Optional[int],
    adapter_memory_size: Optional[int],
    adapter_spm_memory_size: Optional[int],
    adapter_itm_memory_size: Optional[int],
    adapter_ngram_order: Optional[int],
    adapter_lngram_bits_per_route: Optional[int],
    adapter_topk: Optional[int],
    adapter_use_alpha_layer: Optional[bool],
    adapter_alpha_init: Optional[float],
    adapter_fixed_context_gate: Optional[float],
    adapter_disable_conv: Optional[bool],
    adapter_use_sid_span: Optional[bool],
    adapter_sid_token_start: Optional[int],
    adapter_sid_token_end: Optional[int],
) -> None:
    if adapter_type in (None, "", "none"):
        return

    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        logger.warning(f"Cannot update adapter config because config.json is missing: {config_path}")
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["adapter_type"] = adapter_type
    if adapter_layers is not None:
        config["adapter_layers"] = adapter_layers
    if adapter_hidden_size is not None:
        config["adapter_hidden_size"] = adapter_hidden_size
    if adapter_spm_hidden_size is not None:
        config["adapter_spm_hidden_size"] = adapter_spm_hidden_size
    if adapter_itm_hidden_size is not None:
        config["adapter_itm_hidden_size"] = adapter_itm_hidden_size
    if adapter_memory_size is not None:
        config["adapter_memory_size"] = adapter_memory_size
    if adapter_spm_memory_size is not None:
        config["adapter_spm_memory_size"] = adapter_spm_memory_size
    if adapter_itm_memory_size is not None:
        config["adapter_itm_memory_size"] = adapter_itm_memory_size
    if adapter_ngram_order is not None:
        config["adapter_ngram_order"] = adapter_ngram_order
    if adapter_lngram_bits_per_route is not None:
        config["adapter_lngram_bits_per_route"] = adapter_lngram_bits_per_route
    if adapter_topk is not None:
        config["adapter_topk"] = adapter_topk
    if adapter_use_alpha_layer is not None:
        config["adapter_use_alpha_layer"] = adapter_use_alpha_layer
    if adapter_alpha_init is not None:
        config["adapter_alpha_init"] = adapter_alpha_init
    if adapter_fixed_context_gate is not None:
        config["adapter_fixed_context_gate"] = adapter_fixed_context_gate
    if adapter_disable_conv is not None:
        config["adapter_use_conv"] = not adapter_disable_conv
    if adapter_use_sid_span is not None:
        config["adapter_use_sid_span"] = adapter_use_sid_span
    if adapter_sid_token_start is not None:
        config["adapter_sid_token_start"] = adapter_sid_token_start
    if adapter_sid_token_end is not None:
        config["adapter_sid_token_end"] = adapter_sid_token_end
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Updated adapter config in {config_path}: adapter_type={adapter_type}")


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints (DCP or .pth) to HuggingFace format"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to DCP checkpoint directory or .pth file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted HuggingFace model"
    )

    parser.add_argument(
        "--source_hf_model_path",
        type=str,
        default=None,
        help="Path to original HuggingFace model to copy config files from (optional)"
    )

    parser.add_argument(
        "--use_safetensor",
        action="store_true",
        default=True,
        help="Use safetensors format (default: True)"
    )

    parser.add_argument(
        "--no_safetensor",
        dest="use_safetensor",
        action="store_false",
        help="Use .bin format instead of safetensors"
    )

    parser.add_argument(
        "--max_gb_per_shard",
        type=int,
        default=DEFAULT_MAX_GB_PER_SHARD,
        help=f"Maximum size per shard in GB (default: {DEFAULT_MAX_GB_PER_SHARD})"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["fp32", "fp16", "bf16"],
        help=f"Data type for conversion (default: {DEFAULT_DTYPE})"
    )
    parser.add_argument("--adapter_type", type=str, default=None,
                        help="Memory adapter type to write into converted config")
    parser.add_argument("--houlsby_adapter", action="store_true",
                        help="Enable the Houlsby bottleneck adapter in converted config")
    parser.add_argument("--houlsby_adapter_bottleneck", type=int, default=256,
                        help="Houlsby bottleneck width to write into converted config")
    parser.add_argument("--adapter_layers", type=str, default=None,
                        help="Comma-separated 0-indexed adapter layers to write into converted config")
    parser.add_argument("--adapter_hidden_size", type=int, default=None,
                        help="Memory adapter hidden size to write into converted config")
    parser.add_argument("--adapter_spm_hidden_size", type=int, default=None,
                        help="SPM embedding width to write into converted config")
    parser.add_argument("--adapter_itm_hidden_size", type=int, default=None,
                        help="ITM embedding width to write into converted config")
    parser.add_argument("--adapter_memory_size", type=int, default=None,
                        help="Memory adapter memory size to write into converted config")
    parser.add_argument("--adapter_spm_memory_size", type=int, default=None,
                        help="SPM memory size to write into converted config")
    parser.add_argument("--adapter_itm_memory_size", type=int, default=None,
                        help="ITM memory size to write into converted config")
    parser.add_argument("--adapter_ngram_order", type=int, default=None,
                        help="Memory adapter ngram order to write into converted config")
    parser.add_argument("--adapter_lngram_bits_per_route", type=int, default=None,
                        help="Lngram bits per route to write into converted config")
    parser.add_argument("--adapter_topk", type=int, default=None,
                        help="Memory adapter top-k to write into converted config")
    parser.add_argument("--adapter_use_alpha_layer", action="store_true", default=None,
                        help="Write learnable final Engram residual alpha configuration")
    parser.add_argument("--adapter_alpha_init", type=float, default=None,
                        help="Write Engram alpha initial value into converted config")
    parser.add_argument("--adapter_fixed_context_gate", type=float, default=None,
                        help="Write a fixed Engram context gate into converted config")
    parser.add_argument("--adapter_disable_conv", action="store_true", default=None,
                        help="Write disabled Engram causal convolution configuration")
    parser.add_argument("--adapter_use_sid_span", action="store_true", default=None,
                        help="Write SID-span-only Engram routing configuration")
    parser.add_argument("--adapter_sid_token_start", type=int, default=None,
                        help="Write inclusive first SID token id for Engram routing")
    parser.add_argument("--adapter_sid_token_end", type=int, default=None,
                        help="Write inclusive last SID token id for Engram routing")
    parser.add_argument("--bundle_remote_code", action="store_true",
                        help="Copy local remote-code files into the converted checkpoint directory")
    parser.add_argument("--merge_lora", action="store_true",
                        help="Merge LoRA weights into base Linear weights before writing HF shards")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="LoRA rank used for merge")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha used for merge")

    return parser


def restore_structure_config(checkpoint_path, output_dir):
    """Use the actual training-run config, not base-checkpoint adapter defaults."""
    checkpoint_path = Path(checkpoint_path).resolve()
    for parent in [checkpoint_path] + list(checkpoint_path.parents)[:3]:
        source = parent / "structure_adapter_config.json"
        if source.is_file():
            config_file = Path(output_dir) / "config.json"
            config = json.loads(config_file.read_text())
            config.update(json.loads(source.read_text()))
            config_file.write_text(json.dumps(config, indent=2) + "\n")
            return True
    return False


def validate_itm_aux_export(output_dir):
    """Fail closed if exported prior/auxiliary config and weights disagree.

    Inspect safetensor headers instead of materializing the full model. Training
    sidecars remain authoritative; conversion never invents or drops a head.
    """
    root = Path(output_dir)
    config = json.loads((root / 'config.json').read_text())
    weight = float(config.get('adapter_itm_aux_loss_weight', 0.))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid exported ITM auxiliary loss weight')
    enabled = config.get('adapter_itm_aux_head', weight > 0)
    if weight > 0 and not enabled:
        raise ValueError('Positive auxiliary weight requires a saved next-a head')
    if enabled and (config.get('adapter_type') != 'engram_structure'
                    or not config.get('adapter_use_itm', True)
                    or config.get('adapter_itm_mode') != 'exact'):
        raise ValueError('Auxiliary head requires enabled exact ITM')
    index_files = [root / 'model.safetensors.index.json', root / 'pytorch_model.bin.index.json',
                   root / 'model.bin.index.json']
    index = next((p for p in index_files if p.is_file()), None)
    if index:
        weight_map = json.loads(index.read_text())['weight_map']
        files = [root / name for name in sorted(set(weight_map.values()))]
    else:
        files = [p for p in (root / 'model.safetensors', root / 'pytorch_model.bin') if p.is_file()]
    shapes = {}
    prior_shapes = {}
    prior_mode = config.get('adapter_itm_mode') in ('successor_prior', 'successor_prior_learned')
    learned_prior_mode = config.get('adapter_itm_mode') == 'successor_prior_learned'
    def capture_prior(name, shape):
        if (prior_mode and '.memory_adapter.itm.' in name) or '.itm.prior_proj.' in name:
            prior_shapes[name] = tuple(shape)
    for file in files:
        if file.suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(file, framework='pt', device='cpu') as handle:
                for name in handle.keys():
                    capture_prior(name, handle.get_slice(name).get_shape())
                    if name in prior_shapes and index and weight_map.get(name) != file.name:
                        raise ValueError(f'Prior weight missing or inconsistent in shard index: {name}')
                    if '.itm.next_a_head.' in name:
                        if index and weight_map.get(name) != file.name:
                            raise ValueError(f'Auxiliary head missing or inconsistent in shard index: {name}')
                        shapes[name] = tuple(handle.get_slice(name).get_shape())
        else:
            tensors = torch.load(file, map_location='cpu', weights_only=True)
            for name, tensor in tensors.items():capture_prior(name, tensor.shape)
            shapes.update({k: tuple(v.shape) for k, v in tensors.items() if '.itm.next_a_head.' in k})
    expected = {}
    if enabled:
        layers = config.get('adapter_layers')
        if layers in (None, '', 'all'):
            layers = range(config['num_hidden_layers'])
        elif isinstance(layers, str):
            layers = [int(v.strip()) for v in layers.split(',') if v.strip()]
        ranges = config['adapter_sid_code_ranges']
        classes = ranges[0][1] - ranges[0][0] + 1
        itm_dim = (config.get('adapter_itm_hidden_size') or config.get('adapter_hidden_size')
                   or config['hidden_size'])
        width = itm_dim * config.get('adapter_topk', 4)
        expected = {f'model.layers.{i}.memory_adapter.itm.next_a_head.weight': (classes, width) for i in layers}
        if not expected:
            raise ValueError('Auxiliary head enabled but no ITM layers selected')
    if shapes != expected:
        raise ValueError(f'Exported next-a head weights/config mismatch: expected {expected}, found {shapes}')
    expected_prior = {}
    if prior_mode:
        from types import SimpleNamespace
        from onerec_llm.models.memory_adapter.structure import validate_successor_prior
        validate_successor_prior(SimpleNamespace(**config))
        ranges = config['adapter_sid_code_ranges']
        classes = ranges[0][1] - ranges[0][0] + 1
        layers = config.get('adapter_layers')
        if layers in (None, '', 'all'):
            layers = range(config['num_hidden_layers'])
        elif isinstance(layers, str):
            layers = [int(v.strip()) for v in layers.split(',') if v.strip()]
        if config.get('adapter_use_itm', True):
            hidden = config['hidden_size']
            dim = (config.get('adapter_itm_hidden_size') or config.get('adapter_hidden_size') or hidden)
            if learned_prior_mode:
                width = dim * config.get('adapter_topk', 4)
                tensors = {f'memories.{h}.weight':(classes, dim) for h in range(config.get('adapter_topk', 4))}
                tensors.update({'key_proj.weight':(hidden,width), 'value_proj.weight':(hidden,width),
                                'query_norm.weight':(hidden,), 'key_norm.weight':(hidden,)})
            else:
                tensors = {'prior_proj.weight':(dim,hidden), 'key_proj.weight':(hidden,dim),
                           'value_proj.weight':(hidden,dim), 'query_norm.weight':(hidden,), 'key_norm.weight':(hidden,)}
            if config.get('adapter_use_alpha_layer', False):tensors['alpha_layer']=(1,)
            expected_prior = {f'model.layers.{i}.memory_adapter.itm.{k}':v for i in layers for k,v in tensors.items()}
            if not expected_prior:
                raise ValueError('Successor prior enabled but no adapter layers selected')
    if prior_shapes != expected_prior:
        raise ValueError(f'Exported successor prior weights/config mismatch: expected {expected_prior}, found {prior_shapes}')


def main() -> None:
    """Main entry point for the script."""
    parser = get_argument_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Auto-detect input type: .pth file or DCP checkpoint directory
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pth":
        logger.info(f"Detected PTH file: {checkpoint_path}")
        pth_to_hf_format(
            pth_file_path=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype,
            merge_lora=args.merge_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
    elif checkpoint_path.is_dir():
        logger.info(f"Detected DCP checkpoint directory: {checkpoint_path}")
        dcp_to_hf_format(
            dcp_checkpoint_dir=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype,
            merge_lora=args.merge_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
    else:
        raise ValueError(
            f"Invalid checkpoint path: {checkpoint_path}. "
            "Expected either a .pth file or a DCP checkpoint directory."
        )

    # Copy config files if source model path is provided
    if args.source_hf_model_path:
        logger.info(f"Copying config files from {args.source_hf_model_path} to {args.output_dir}")
        copy_hf_config_files(
            source_hf_model_path=args.source_hf_model_path,
            output_dir=args.output_dir
        )
        update_adapter_config(
            output_dir=args.output_dir,
            adapter_type=args.adapter_type,
            adapter_layers=args.adapter_layers,
            adapter_hidden_size=args.adapter_hidden_size,
            adapter_spm_hidden_size=args.adapter_spm_hidden_size,
            adapter_itm_hidden_size=args.adapter_itm_hidden_size,
            adapter_memory_size=args.adapter_memory_size,
            adapter_spm_memory_size=args.adapter_spm_memory_size,
            adapter_itm_memory_size=args.adapter_itm_memory_size,
            adapter_ngram_order=args.adapter_ngram_order,
            adapter_lngram_bits_per_route=args.adapter_lngram_bits_per_route,
            adapter_topk=args.adapter_topk,
            adapter_use_alpha_layer=args.adapter_use_alpha_layer,
            adapter_alpha_init=args.adapter_alpha_init,
            adapter_fixed_context_gate=args.adapter_fixed_context_gate,
            adapter_disable_conv=args.adapter_disable_conv,
            adapter_use_sid_span=args.adapter_use_sid_span,
            adapter_sid_token_start=args.adapter_sid_token_start,
            adapter_sid_token_end=args.adapter_sid_token_end,
        )
        restore_structure_config(checkpoint_path, args.output_dir)
        final_config = json.loads((Path(args.output_dir) / "config.json").read_text())
        if args.houlsby_adapter:
            final_config["houlsby_adapter"] = True
            final_config["houlsby_adapter_bottleneck"] = args.houlsby_adapter_bottleneck
            (Path(args.output_dir) / "config.json").write_text(json.dumps(final_config, indent=2) + "\n")
        is_structure = final_config.get("adapter_type") in ("engram_structure", "lngram_structure")
        if is_structure and not final_config.get("adapter_sid_code_ranges"):
            raise ValueError("Missing trained structure_adapter_config.json; cannot safely export structured model")
        validate_itm_aux_export(args.output_dir)
        if args.bundle_remote_code:
            copy_remote_code_bundle(args.output_dir)
            update_remote_code_config(args.output_dir)

    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
