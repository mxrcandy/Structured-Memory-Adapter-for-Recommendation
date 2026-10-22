"""lm-eval token requests executed by the recommendation Ray/VllmWorker path."""
from __future__ import annotations

import importlib
import faulthandler
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import ray


def expose_benchmark():
    # Expose reusable modules without importing the recommendation evaluator.
    root = Path(__file__).resolve().parents[1]
    if "benchmark" not in sys.modules:
        package = types.ModuleType("benchmark")
        package.__path__ = [str(root / "benchmark")]
        sys.modules["benchmark"] = package


def export_benchmark_path():
    """Expose ``benchmark`` to vLLM's fresh architecture-inspection process."""
    root = Path(__file__).resolve().parents[1]
    pythonpath = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    root_str = str(root)
    if root_str not in pythonpath:
        os.environ["PYTHONPATH"] = os.pathsep.join([root_str, *pythonpath])


class RecommendationReplica:
    def __init__(self, worker_id, model_args):
        expose_benchmark()
        # vLLM inspects custom architectures in a subprocess.  Set this only
        # inside the Ray actor so the driver does not shadow site-packages' lm_eval.
        export_benchmark_path()
        worker_class = importlib.import_module(
            "benchmarks.scripts.ray-vllm.utils.generator"
        ).VllmWorker
        args = dict(model_args)
        model_path = args.pop("model")
        args.pop("distributed_executor_backend", None)
        args.pop("data_parallel_size", None)
        # Use Ray's assigned visibility, never physical indices from the driver.
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        print(f"[OpenOneRec DP] worker={worker_id} assigned GPUs={gpu_ids}", flush=True)
        self.worker = worker_class(
            worker_id=worker_id, model_path=model_path, gpu_ids=gpu_ids,
            enable_optimizations=False, **args,
        )

    def ready(self):
        return self.worker.ready()

    def generate(self, *args):
        return self.worker.generate_token_requests(*args)


def _data_parallel_generate(self, requests=None, generate=False, max_tokens=None, stop=None, **kwargs):
    from vllm import SamplingParams

    if generate:
        kwargs = self.modify_gen_kwargs(kwargs)
        sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
    else:
        sampling_params = SamplingParams(
            temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
        )

    if not getattr(self, "data_parallel_size", 1) > 1:
        return self.model.generate(
            prompt_token_ids=requests,
            sampling_params=sampling_params,
            use_tqdm=True if self.batch_size == "auto" else False,
            lora_request=self.lora_request,
        )

    if not hasattr(self, "_openonerec_dp_workers"):
        print(f"[OpenOneRec DP] starting {self.data_parallel_size} single-GPU replicas", flush=True)
        remote_class = ray.remote(num_gpus=1)(RecommendationReplica)
        self._openonerec_dp_workers = []
        for i in range(self.data_parallel_size):
            cache = os.path.join(os.environ.get("HF_MODULES_CACHE", "/tmp/openonerec_hf_modules"), f"worker_{i}")
            worker = remote_class.options(runtime_env={"env_vars": {"HF_MODULES_CACHE": cache}}).remote(i, self.model_args)
            self._openonerec_dp_workers.append(worker)
        ready = ray.get([w.ready.remote() for w in self._openonerec_dp_workers], timeout=float(os.environ.get("GENERAL_WORKER_START_TIMEOUT", "600")))
        print(f"[OpenOneRec DP] workers ready: {ready}", flush=True)

    request_shards = [list(requests[i::self.data_parallel_size]) for i in range(self.data_parallel_size)]
    futures = [
        worker.generate.remote(shard, sampling_params, self.lora_request)
        for worker, shard in zip(self._openonerec_dp_workers, request_shards)
        if shard
    ]
    shard_results = ray.get(futures)
    for shard, results in zip((s for s in request_shards if s), shard_results):
        if len(results) != len(shard):
            raise RuntimeError("DP worker response count does not match its shard")
    # Reconstruct the original interleaved request order.
    flattened = []
    for index in range(len(requests)):
        shard_index, offset = divmod(index, self.data_parallel_size)
        flattened.append(shard_results[offset][shard_index])
    if len(flattened) != len(requests):
        raise RuntimeError("DP response count does not match requests")
    return flattened


def install() -> None:
    from lm_eval.models.vllm_causallms import VLLM

    VLLM._model_generate = _data_parallel_generate


def main() -> int:
    expose_benchmark()
    from benchmark.base_generator import RayMixin

    if not ray.is_initialized():
        requested_gpus = int(os.environ.get("DATA_PARALLEL_SIZE", "1"))
        context = SimpleNamespace(ray_address="local", num_gpus=requested_gpus)
        print(f"[OpenOneRec DP] RayMixin init: GPUs={requested_gpus}, RAY_NUM_CPUS={os.environ.get('RAY_NUM_CPUS')}, temp={os.environ.get('RAY_TEMP_DIR')}", flush=True)
        faulthandler.dump_traceback_later(120, repeat=True)
        try:
            RayMixin._initialize_ray_cluster(context)
        finally:
            faulthandler.cancel_dump_traceback_later()
        print(f"[OpenOneRec DP] Ray initialized; cluster resources={ray.cluster_resources()}", flush=True)
        available_gpus = int(ray.cluster_resources().get("GPU", 0))
        if available_gpus < requested_gpus:
            raise RuntimeError(
                f"Ray detected {available_gpus} GPUs, but DATA_PARALLEL_SIZE={requested_gpus}. "
                "Check CUDA_VISIBLE_DEVICES and Slurm GPU allocation."
            )
    install()
    from benchmarks.lm_eval.protocol import install as install_protocol
    install_protocol()
    from lm_eval.__main__ import cli_evaluate

    try:
        return int(cli_evaluate() or 0)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
