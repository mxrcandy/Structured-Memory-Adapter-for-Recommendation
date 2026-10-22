import json
import subprocess
import sys
from pathlib import Path

from benchmarks.lm_eval.wrapper import build_command, detect_adapter_type, detect_houlsby_adapter, load_suite


def test_suite_loads_tasks_and_defaults(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: smoke\ntasks:\n  - gsm8k\n  - mmlu_pro\nnum_fewshot: 0\nseed: 42\n",
        encoding="utf-8",
    )

    config = load_suite(suite)

    assert config["name"] == "smoke"
    assert config["tasks"] == ["gsm8k", "mmlu_pro"]
    assert config["num_fewshot"] == 0
    assert config["seed"] == 42


def test_build_command_contains_model_and_reproducibility_flags(tmp_path):
    command = build_command(
        model_path="/models/converted",
        suite={"tasks": ["gsm8k", "ifeval"], "num_fewshot": 0, "seed": 42},
        output_path=tmp_path / "results.json",
        model_backend="vllm",
        enable_thinking=False,
        adapter_enabled=False,
        limit=3,
    )

    assert command[:3] == [sys.executable, "-m", "benchmarks.lm_eval.protocol"]
    assert "--tasks" in command
    assert command[command.index("--tasks") + 1] == "gsm8k,ifeval"
    model_args = command[command.index("--model_args") + 1]
    assert "pretrained=/models/converted" in model_args
    assert "trust_remote_code=True" in model_args
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--limit") + 1] == "3"
    assert command[command.index("--output_path") + 1] == str(tmp_path / "results.json")


def test_wrapper_module_is_importable_without_starting_evaluation():
    result = subprocess.run(
        ["python", "-c", "import benchmarks.lm_eval.wrapper"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_detect_adapter_type_reads_checkpoint_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"adapter_type": "engram_structure"}), encoding="utf-8"
    )

    assert detect_adapter_type(tmp_path) == "engram_structure"


def test_houlsby_checkpoint_uses_default_vllm_model_impl(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"houlsby_adapter": True}), encoding="utf-8")
    assert detect_houlsby_adapter(tmp_path)
    command = build_command(str(tmp_path), {"tasks": ["gsm8k"], "num_fewshot": 0, "seed": 1}, tmp_path / "out.json")
    model_args = command[command.index("--model_args") + 1]
    assert "model_impl=" not in model_args
    disabled = build_command(str(tmp_path), {"tasks": ["gsm8k"], "num_fewshot": 0, "seed": 1}, tmp_path / "off.json", adapter_enabled=False)
    assert "model_impl=" not in disabled[disabled.index("--model_args") + 1]


def test_structure_model_uses_local_launcher_without_changing_vllm_backend(tmp_path):
    command = build_command(
        model_path="/models/converted",
        suite={"tasks": ["gsm8k"], "num_fewshot": 0, "seed": 42},
        output_path=tmp_path / "results.json",
        adapter_type="lngram_structure",
    )

    assert command[0:3] == [sys.executable, "-m", "benchmarks.lm_eval.structure_launcher"]
    assert command[command.index("--model") + 1] == "vllm"


def test_ordinary_adapter_uses_shared_protocol_launcher(tmp_path):
    command = build_command(
        model_path="/models/converted",
        suite={"tasks": ["gsm8k"], "num_fewshot": 0, "seed": 42},
        output_path=tmp_path / "results.json",
        adapter_type="engram",
    )

    assert command[:3] == [sys.executable, "-m", "benchmarks.lm_eval.protocol"]


def test_structure_launcher_supplies_existing_bridge_kwargs():
    from benchmarks.lm_eval.structure_launcher import structure_model_kwargs

    kwargs = structure_model_kwargs()

    assert kwargs["worker_cls"].__name__ == "StructureWorker"
    assert kwargs["worker_cls"].__module__ == "benchmarks.lm_eval.structure_bridge"
    assert kwargs["hf_overrides"] == {
        "architectures": ["OpenOneRecStructuredTransformersModel"]
    }
    assert kwargs["enforce_eager"] is True
    assert kwargs["disable_async_output_proc"] is True
