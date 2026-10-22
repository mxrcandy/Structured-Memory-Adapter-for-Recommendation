"""Import-safe facade for the existing recommendation structure bridge.

vLLM inspects registered model classes in a fresh subprocess. Importing the
top-level recommendation package there would pull optional console dependencies,
so this facade exposes only the bridge module and keeps its implementation
unchanged.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

from vllm import ModelRegistry


def _load_bridge_module():
    benchmark_dir = Path(__file__).resolve().parents[1] / "benchmark"
    if "benchmark" not in sys.modules:
        package = types.ModuleType("benchmark")
        package.__path__ = [str(benchmark_dir)]
        package.__package__ = "benchmark"
        sys.modules["benchmark"] = package
    from benchmark import vllm_structure

    return vllm_structure


_bridge = _load_bridge_module()
ARCHITECTURE = _bridge.ARCHITECTURE


class StructuredTransformersModel(_bridge.StructuredTransformersModel):
    pass


def register_structure_model() -> None:
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE,
            "benchmarks.lm_eval.structure_bridge:StructuredTransformersModel",
        )


class StructureWorker(_bridge.StructureWorker):
    def __init__(self, *args, **kwargs):
        register_structure_model()
        super().__init__(*args, **kwargs)


def bridge_kwargs():
    register_structure_model()
    return {
        "hf_overrides": {"architectures": [ARCHITECTURE]},
        "worker_cls": StructureWorker,
        "enforce_eager": True,
        "disable_async_output_proc": True,
    }
