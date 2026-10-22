import os
import sys
from types import SimpleNamespace

import pytest

from benchmarks.lm_eval import data_parallel_launcher as dp


@pytest.mark.parametrize("count", [0, 1, 3, 4, 7, 13])
def test_shards_keep_order_and_reuse_workers(monkeypatch, count):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(dp.ray, "get", lambda refs: refs)
    shards = []
    workers = [SimpleNamespace(generate=SimpleNamespace(remote=lambda req, params, lora: shards.append(req) or list(req))) for _ in range(4)]
    model = SimpleNamespace(data_parallel_size=4, _openonerec_dp_workers=workers, lora_request=None)
    requests = list(range(count))
    assert dp._data_parallel_generate(model, requests) == requests
    assert sorted(x for shard in shards for x in shard) == requests
    assert model._openonerec_dp_workers is workers


def test_replica_uses_recommendation_worker_and_assigned_gpu(monkeypatch):
    captured = {}
    class Worker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def generate_token_requests(self, *args):
            return args
        def ready(self):
            return True
    monkeypatch.setattr(dp, "expose_benchmark", lambda: None)
    monkeypatch.setattr(dp.importlib, "import_module", lambda name: SimpleNamespace(VllmWorker=Worker))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned-uuid")
    original = dict(model="/model", tensor_parallel_size=1, distributed_executor_backend="ray", seed=42)
    replica = dp.RecommendationReplica(2, original)
    assert captured["gpu_ids"] == ["GPU-assigned-uuid"]
    assert captured["model_path"] == "/model"
    assert captured["seed"] == 42
    assert "distributed_executor_backend" not in captured
    assert original["distributed_executor_backend"] == "ray"
    tokens, params = [[1, 2]], object()
    assert replica.generate(tokens, params, None) == (tokens, params, None)


def test_export_benchmark_path_propagates_to_subprocess(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    dp.export_benchmark_path()

    expected = str(dp.Path(__file__).resolve().parents[1])
    assert expected in os.environ["PYTHONPATH"].split(os.pathsep)


def test_launcher_calls_shared_ray_initialization(monkeypatch):
    calls = []
    class Mixin:
        def _initialize_ray_cluster(self):
            calls.append((self.ray_address, self.num_gpus))
    monkeypatch.setattr(dp, "expose_benchmark", lambda: None)
    monkeypatch.setattr(dp, "install", lambda: None)
    monkeypatch.setitem(sys.modules, "benchmark.base_generator", SimpleNamespace(RayMixin=Mixin))
    monkeypatch.setitem(sys.modules, "lm_eval.__main__", SimpleNamespace(cli_evaluate=lambda: 0))
    monkeypatch.setattr(dp.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(dp.ray, "cluster_resources", lambda: {"GPU": 4})
    monkeypatch.setattr(dp.ray, "shutdown", lambda: calls.append("shutdown"))
    monkeypatch.setenv("DATA_PARALLEL_SIZE", "4")
    assert dp.main() == 0
    assert calls == [("local", 4), "shutdown"]
