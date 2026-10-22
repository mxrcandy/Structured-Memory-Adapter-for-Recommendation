"""Bootstrap lm-eval with OpenOneRec's existing structured vLLM bridge."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

# Import the installed harness before exposing ``benchmarks/`` as a top-level
# package root; that directory also contains this integration package named
# ``lm_eval``.
from lm_eval.__main__ import cli_evaluate


def _expose_benchmark_package() -> None:
    benchmark_root_path = Path(__file__).resolve().parents[1]
    benchmark_root = str(benchmark_root_path)
    if benchmark_root not in sys.path:
        sys.path.append(benchmark_root)
    python_path = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in python_path.split(os.pathsep) if entry]
    if benchmark_root not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([*entries, benchmark_root])
    # The recommendation package initializer imports console-only dependencies
    # which are deliberately absent from the lm-eval environment. Expose just
    # its module path so importing the reusable bridge does not initialize the
    # recommendation evaluator.
    if "benchmark" not in sys.modules:
        package = types.ModuleType("benchmark")
        package.__path__ = [str(benchmark_root_path / "benchmark")]
        package.__package__ = "benchmark"
        sys.modules["benchmark"] = package


def structure_model_kwargs() -> dict[str, Any]:
    _expose_benchmark_package()
    from benchmarks.lm_eval.structure_bridge import bridge_kwargs

    return bridge_kwargs()


def install_structure_backend() -> None:
    from lm_eval.api import registry
    from lm_eval.models.vllm_causallms import VLLM

    bridge_kwargs = structure_model_kwargs()

    class StructureVLLM(VLLM):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            overrides = dict(kwargs.get("hf_overrides") or {})
            overrides.update(bridge_kwargs["hf_overrides"])
            kwargs.update(bridge_kwargs)
            kwargs["hf_overrides"] = overrides
            super().__init__(*args, **kwargs)

    registry.MODEL_REGISTRY["vllm"] = StructureVLLM


def main() -> int:
    install_structure_backend()
    from benchmarks.lm_eval.protocol import install as install_protocol
    install_protocol()
    result = cli_evaluate()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
