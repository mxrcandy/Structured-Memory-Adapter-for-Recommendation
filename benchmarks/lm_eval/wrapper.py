"""Run lm-evaluation-harness with OneRec reproducibility metadata."""
from __future__ import annotations
import argparse
import atexit
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from typing import Any
import yaml
from benchmarks.lm_eval.protocol import GSM_PROMPT

SUITE_DIR = Path(__file__).resolve().parent / "suites"
STRUCTURE_ADAPTERS = {"engram_structure", "lngram_structure"}


def prepare_short_model_path(model_path: str | Path, output_dir: Path) -> tuple[str, Path | None]:
    """Create a per-job short symlink for lm-eval's sample output paths.

    lm-eval derives a directory name from ``pretrained`` when ``--log_samples``
    is enabled.  Passing the full checkpoint path can exceed NAME_MAX on the
    cluster.  The link root is configurable and defaults next to the output,
    so no machine-specific absolute path is embedded in the code.
    """
    source = Path(model_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {source}")
    job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    root_value = os.environ.get("OPENONEREC_MODEL_LINK_ROOT")
    link_root = Path(root_value).expanduser() if root_value else Path.home() / ".openonerec" / "model_links"
    link_root.mkdir(parents=True, exist_ok=True)
    link_path = link_root / job_id
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == source:
            return str(link_path), link_path
        raise FileExistsError(f"Short model link already exists: {link_path}")
    link_path.symlink_to(source, target_is_directory=True)
    atexit.register(lambda: link_path.unlink(missing_ok=True))
    return str(link_path), link_path


def detect_adapter_type(model_path: str | Path) -> str | None:
    config_path = Path(model_path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    adapter_type = config.get("adapter_type")
    return adapter_type if isinstance(adapter_type, str) else None

def detect_houlsby_adapter(model_path: str | Path) -> bool:
    config_path = Path(model_path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("houlsby_adapter") is True

def load_suite(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f"Suite must define a non-empty string list: {path}")
    config["name"] = str(config.get("name") or path.stem)
    config["num_fewshot"] = int(config.get("num_fewshot", 0))
    config["seed"] = config.get("seed", 42)
    return config

def build_command(model_path: str, suite: dict[str, Any], output_path: Path, model_backend: str = "vllm", enable_thinking: bool = False, adapter_enabled: bool = True, limit: int | float | None = None, adapter_type: str | None = None) -> list[str]:
    # Qwen3 thinking is a tokenizer-template option, not a vLLM engine argument.
    # The shared protocol installs template control in each local launcher.
    model_args = [f"pretrained={model_path}", "trust_remote_code=True", "dtype=bfloat16", f"gpu_memory_utilization={os.environ.get('GPU_MEMORY_UTILIZATION', '0.8')}"]
    if os.environ.get("TENSOR_PARALLEL_SIZE"):
        model_args.append(f"tensor_parallel_size={os.environ['TENSOR_PARALLEL_SIZE']}")
    if os.environ.get("DATA_PARALLEL_SIZE"):
        model_args.append(f"data_parallel_size={os.environ['DATA_PARALLEL_SIZE']}")
    if adapter_type in STRUCTURE_ADAPTERS:
        command = [sys.executable, "-m", "benchmarks.lm_eval.structure_launcher"]
    else:
        command = [sys.executable, "-m", "benchmarks.lm_eval.protocol"]
    command += ["--model", model_backend, "--model_args", ",".join(model_args), "--tasks", ",".join(suite["tasks"]), "--num_fewshot", str(suite["num_fewshot"]), "--seed", str(suite["seed"]), "--trust_remote_code", "--apply_chat_template", "--output_path", str(output_path), "--log_samples"]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    command.extend(["--batch_size", "auto"])
    return command


def run_data_parallel(args: argparse.Namespace, suite: dict[str, Any], output_dir: Path) -> int:
    if args.data_parallel_size < 2:
        raise ValueError("data parallel mode requires --data-parallel-size >= 2")
    # The local launcher delegates token requests to recommendation workers.
    previous_tp = os.environ.get("TENSOR_PARALLEL_SIZE")
    os.environ["TENSOR_PARALLEL_SIZE"] = "1"
    os.environ["DATA_PARALLEL_SIZE"] = str(args.data_parallel_size)
    launcher = "benchmarks.lm_eval.data_parallel_launcher"
    command = build_command(
        args.model_path, suite, output_dir / "results.json",
        enable_thinking=args.enable_thinking, adapter_enabled=args.adapter_enabled == "true",
        limit=args.limit, adapter_type=detect_adapter_type(args.model_path),
    )
    # The launcher is a repo-local lm-eval bootstrap that patches vLLM's
    # nested-Ray data-parallel implementation before invoking cli_evaluate.
    if command[:2] == [sys.executable, "-m"]:
        command[:3] = [sys.executable, "-m", launcher]
    else:
        command = [sys.executable, "-m", launcher, *command[1:]]
    if previous_tp is None:
        os.environ.pop("TENSOR_PARALLEL_SIZE", None)
    else:
        os.environ["TENSOR_PARALLEL_SIZE"] = previous_tp
    versions = {package: (version(package) if _package_available(package) else "unknown") for package in ("lm-eval", "torch", "transformers", "vllm", "datasets")}
    manifest = {"backend": "lm_eval", "suite": suite["name"], "tasks": suite["tasks"], "model_path": str(Path(args.model_path).resolve()), "parallel_mode": "data", "data_parallel_size": args.data_parallel_size, "command": shlex.join(command), "versions": versions}
    manifest.update(protocol_metadata(args))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(shlex.join(command), flush=True)
    return subprocess.run(command, check=False, cwd=Path(__file__).resolve().parents[2]).returncode


def protocol_metadata(args):
    return {"gsm8k_prompt": GSM_PROMPT,
            "enable_thinking": args.enable_thinking,
            "thinking_control_applied": True,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "num_fewshot": 0, "adapter_enabled": args.adapter_enabled == "true",
            "adapter_type": detect_adapter_type(args.model_path)}


def _package_available(package: str) -> bool:
    try:
        version(package)
        return True
    except Exception:
        return False


def _run_task_worker(task: str, args: argparse.Namespace, suite: dict[str, Any], gpu_id: str, worker_dir: Path) -> tuple[str, int]:
    worker_suite = dict(suite, tasks=[task])
    worker_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        args.model_path, worker_suite, worker_dir / "results.json",
        enable_thinking=args.enable_thinking, adapter_enabled=args.adapter_enabled == "true",
        limit=args.limit, adapter_type=detect_adapter_type(args.model_path),
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["TENSOR_PARALLEL_SIZE"] = "1"
    print(f"[parallel worker gpu={gpu_id} task={task}] {shlex.join(command)}", flush=True)
    return task, subprocess.run(command, check=False, cwd=Path(__file__).resolve().parents[2], env=env).returncode


def run_task_parallel(args: argparse.Namespace, suite: dict[str, Any], output_dir: Path) -> int:
    default_workers = args.num_workers or int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    gpu_ids = args.gpu_ids.split() if args.gpu_ids else [str(i) for i in range(default_workers)]
    if args.num_workers:
        gpu_ids = gpu_ids[:args.num_workers]
    if not gpu_ids:
        raise ValueError("parallel mode requires at least one GPU id")
    if len(gpu_ids) < len(suite["tasks"]):
        print(f"Using {len(gpu_ids)} workers for {len(suite['tasks'])} tasks; tasks will run in waves.", flush=True)
    worker_root = output_dir / "parallel_tasks"
    worker_root.mkdir(parents=True, exist_ok=True)
    failures = []
    # Run at most one model process per GPU. When the suite has more tasks than
    # GPUs, process them in waves so two vLLM instances never contend for one
    # CUDA device.
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        tasks = list(suite["tasks"])
        for offset in range(0, len(tasks), len(gpu_ids)):
            wave = tasks[offset:offset + len(gpu_ids)]
            futures = [pool.submit(_run_task_worker, task, args, suite, gpu_ids[i], worker_root / task) for i, task in enumerate(wave)]
            for future in as_completed(futures):
                task, code = future.result()
                if code:
                    failures.append((task, code))
    if failures:
        print(f"Parallel evaluation failed: {failures}", file=sys.stderr)
        return 1
    merged = {"results": {}, "configs": {}, "versions": {}}
    for task in suite["tasks"]:
        result_path = worker_root / task / "results.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("results", "configs"):
            merged[key].update(payload.get(key, {}))
        merged["versions"].update(payload.get("versions", {}))
    merged["config"] = {"suite": suite["name"], "tasks": suite["tasks"], "parallel_mode": "task", "gpu_ids": gpu_ids}
    (output_dir / "results.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0

def prepare_output_dir(output_dir: Path) -> None:
    """Replace this run's artifacts without touching unrelated files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.json", "parallel_tasks", "manifest.json", "diagnostics.json",
                 "generation_diagnostics.jsonl", "prompt_preview.txt"):
        path = output_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--suite", default="stage4_core")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--adapter-enabled", choices=("true", "false"), default="true")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--parallel-mode", choices=("tensor", "data", "task"), default="data")
    parser.add_argument("--data-parallel-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    suite_path = Path(args.suite)
    if not suite_path.exists():
        suite_path = SUITE_DIR / f"{args.suite}.yaml"
    suite = load_suite(suite_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_model_path = str(Path(args.model_path).resolve())
    args.original_model_path = original_model_path
    short_model_path, _model_link = prepare_short_model_path(args.model_path, output_dir)
    args.model_path = short_model_path
    if args.enable_thinking or suite["num_fewshot"] != 0:
        parser.error("General evaluation requires non-thinking and 0-shot")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.parallel_mode == "task":
        parser.error("Use data or tensor mode for the shared evaluation protocol")
    if args.parallel_mode == "data" and args.data_parallel_size < 2:
        parser.error("data parallel mode requires --data-parallel-size >= 2")
    prepare_output_dir(output_dir)
    os.environ["GENERAL_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["GENERAL_PROTOCOL_OUTPUT_DIR"] = str(output_dir.resolve())
    adapter_enabled = args.adapter_enabled == "true"
    os.environ["OPENONEREC_ADAPTER_ENABLED"] = str(adapter_enabled).lower()
    adapter_type = detect_adapter_type(args.model_path)
    if args.parallel_mode == "data":
        code = run_data_parallel(args, suite, output_dir)
        write_diagnostics(args, output_dir, code)
        return code
    if args.parallel_mode == "task":
        os.environ.pop("DATA_PARALLEL_SIZE", None)
        return run_task_parallel(args, suite, output_dir)
    os.environ.pop("DATA_PARALLEL_SIZE", None)
    command = build_command(args.model_path, suite, output_dir / "results.json", enable_thinking=args.enable_thinking, adapter_enabled=adapter_enabled, limit=args.limit, adapter_type=adapter_type)
    versions = {}
    for package in ("lm-eval", "torch", "transformers", "vllm", "datasets"):
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = "unknown"
    manifest = {"backend": "lm_eval", "suite": suite["name"], "tasks": suite["tasks"], "model_path": str(Path(args.model_path).resolve()), "adapter_type": adapter_type, "adapter_enabled": adapter_enabled, "enable_thinking": args.enable_thinking, "command": shlex.join(command), "versions": versions}
    manifest.update(protocol_metadata(args))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(shlex.join(command), flush=True)
    # Run from the repository root: the integration package is named
    # ``benchmarks.lm_eval`` and must not shadow the installed ``lm_eval`` CLI
    # package when the caller's cwd is ``benchmarks/``.
    repo_root = Path(__file__).resolve().parents[2]
    code = subprocess.run(command, check=False, cwd=repo_root).returncode
    write_diagnostics(args, output_dir, code)
    return code


def write_diagnostics(args, output_dir, code):
    if code:
        return
    from collections import Counter
    from benchmarks.lm_eval.protocol import summarize_samples
    counts = Counter()
    total_tokens = 0
    path = output_dir / "generation_diagnostics.jsonl"
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            counts[row["finish_reason"]] += 1
            total_tokens += row["generated_tokens"]
    report = {"protocol": protocol_metadata(args), "samples": summarize_samples(output_dir),
              "finish_reasons": dict(counts), "generated_tokens": total_tokens}
    for result_path in (output_dir / "results.json").rglob("results_*.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["general_protocol"] = protocol_metadata(args)
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Generation finish reasons: {dict(counts)}; diagnostics: {output_dir / 'diagnostics.json'}", flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
