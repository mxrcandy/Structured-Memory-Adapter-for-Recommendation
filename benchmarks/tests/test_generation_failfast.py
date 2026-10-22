"""Generation failures must release workers and must not calculate metrics."""
import ast
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]


class GenerationFailureTests(unittest.TestCase):
    def test_benchmark_propagates_failure(self):
        path = ROOT / 'benchmarks/benchmark/benchmark.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'Benchmark')
        run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        runner = MagicMock(side_effect=KeyError('history_positions'))
        env = dict(Generator=object, GenerationRunner=MagicMock(return_value=runner),
                   BenchmarkTable={'test': {'video': {'splits': ['test']}}},
                   console=MagicMock(), datetime=datetime, head_style='', subhead_style='',
                   err_style='', dim_style='')
        exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), 'exec'), env)
        benchmark = SimpleNamespace(check_generator=lambda _: None, data_loader=None,
                                    benchmark_version='test', task_types=['video'], splits=['test'])
        with self.assertRaisesRegex(RuntimeError, 'Generation failed for video/test') as result:
            env['run'](benchmark, object())
        self.assertIsInstance(result.exception.__cause__, KeyError)

    def test_evaluate_cleans_up_and_skips_metrics(self):
        path = ROOT / 'benchmarks/scripts/ray-vllm/evaluate.py'
        main = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        env = {name: MagicMock() for name in (
            'HfArgumentParser', 'ModelConfig', 'InfrastructureConfig', 'InferenceConfig',
            'GenerationConfig', 'PromptConfig', 'BenchmarkConfig', 'Benchmark',
            'RayVllmGenerator', 'console', 'torch')}
        env.update(warning_style='', success_style='')
        env['HfArgumentParser'].return_value.parse_args_into_dataclasses.return_value = [MagicMock() for _ in range(6)]
        env['Benchmark'].return_value.run.side_effect = RuntimeError('generation failed')
        env['torch'].cuda.is_available.return_value = False
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), 'exec'), env)
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            env['main']()
        env['RayVllmGenerator'].return_value.cleanup.assert_called_once_with()
        env['Benchmark'].evaluate_dev.assert_not_called()


if __name__ == '__main__':
    unittest.main()
