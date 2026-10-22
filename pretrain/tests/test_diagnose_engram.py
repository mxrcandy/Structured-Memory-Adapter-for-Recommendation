import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate.diagnose_engram import (  # noqa: E402
    EngramHashConfig,
    GateStatsAccumulator,
    TokenSequence,
    compare_model_dirs,
    compute_hash_diagnostics,
    compute_single_adapter_gate_stats,
    find_converted_config,
    format_task_row_text,
    token_type_for_id,
)


class TestEngramDiagnostics(unittest.TestCase):
    def test_find_converted_config_under_model_output_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "step264" / "global_step264" / "converted" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"adapter_memory_size": 17}), encoding="utf-8")

            self.assertEqual(find_converted_config(root), config_path)

    def test_hash_diagnostics_counts_collisions_per_table(self):
        config = EngramHashConfig(memory_size=5, ngram_order=2, hash_heads=1)
        sequences = [
            TokenSequence(token_ids=[10, 20, 30, 40], token_types=["text", "sid_item", "sid_item", "text"], source="a"),
            TokenSequence(token_ids=[10, 20, 35, 45], token_types=["text", "sid_item", "sid_item", "text"], source="b"),
        ]

        report = compute_hash_diagnostics(sequences, config)

        table = report["tables"][0]
        self.assertEqual(table["order"], 2)
        self.assertEqual(table["head"], 0)
        self.assertEqual(table["total_lookups"], 8)
        self.assertGreater(table["unique_ngrams"], table["used_buckets"])
        self.assertGreater(table["collision_ngram_rate"], 0.0)
        self.assertIn("mixed_text_sid", table["groups"])
        self.assertGreater(table["groups"]["mixed_text_sid"]["total_lookups"], 0)

    def test_compare_model_dirs_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_a = root / "run_mem5"
            model_b = root / "run_mem11"
            for model_dir, memory_size in [(model_a, 5), (model_b, 11)]:
                config_path = model_dir / "step264" / "global_step264" / "converted" / "config.json"
                config_path.parent.mkdir(parents=True)
                config_path.write_text(
                    json.dumps(
                        {
                            "adapter_type": "engram",
                            "adapter_memory_size": memory_size,
                            "adapter_ngram_order": 3,
                            "adapter_topk": 2,
                        }
                    ),
                    encoding="utf-8",
                )

            output_path = root / "diagnostics" / "report.json"
            report = compare_model_dirs(
                [model_a, model_b],
                sequences=[
                    TokenSequence([10, 20, 30, 40], ["text", "sid_item", "sid_item", "text"], "a"),
                    TokenSequence([20, 30, 40, 50], ["sid_item", "sid_item", "text", "text"], "b"),
                ],
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(report["models"]), 2)
            self.assertEqual(report["models"][0]["configured_memory_size"], 5)
            self.assertEqual(report["models"][1]["configured_memory_size"], 11)
            self.assertIn("provided_sequences", report["models"][0]["datasets"])

    def test_token_type_for_id_distinguishes_sid_items_and_markers(self):
        self.assertEqual(token_type_for_id(101, {101}, {201, 202}), "sid_item")
        self.assertEqual(token_type_for_id(201, {101}, {201, 202}), "sid_marker")
        self.assertEqual(token_type_for_id(999, {101}, {201, 202}), "text")

    def test_format_task_row_text_handles_segments_and_messages(self):
        segment_row = {
            "segments": '[{"type": "text", "text": "abc"}]',
        }
        message_row = {
            "messages": '[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]',
        }

        self.assertEqual(format_task_row_text(segment_row), "abc")
        self.assertEqual(format_task_row_text(message_row), [{"role": "user", "content": "hello"}])

    def test_single_adapter_gate_stats_include_context_gate_and_norm_ratio(self):
        try:
            import torch
            from onerec_llm.models.memory_adapter.engram import EngramMemoryAdapter
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is not installed in this Python environment: {exc}")

        adapter = EngramMemoryAdapter(
            hidden_size=8,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=3,
            hash_heads=2,
        )
        hidden = torch.randn(2, 5, 8)
        input_ids = torch.tensor(
            [
                [10, 20, 30, 40, 50],
                [11, 21, 31, 41, 51],
            ],
            dtype=torch.long,
        )

        stats = compute_single_adapter_gate_stats(
            adapter,
            hidden,
            input_ids=input_ids,
            token_types=[
                "text",
                "sid_marker",
                "sid_item",
                "sid_marker",
                "text",
            ],
        )

        self.assertGreaterEqual(stats["context_gate_min"], 0.0)
        self.assertLessEqual(stats["context_gate_max"], 1.0)
        self.assertIn("delta_hidden_norm_ratio_mean", stats)
        self.assertIn("conv_delta_norm_mean", stats)
        self.assertIn("groups", stats)
        self.assertIn("sid_only", stats["groups"])
        self.assertIn("mixed_text_sid", stats["groups"])

    def test_gate_stats_accumulator_groups_by_layer(self):
        accumulator = GateStatsAccumulator()
        accumulator.add(
            "layer_1",
            {
                "tokens": 10,
                "context_gate_mean": 0.4,
                "context_gate_std": 0.1,
                "context_gate_min": 0.2,
                "context_gate_max": 0.8,
                "context_gate_p95": 0.7,
                "delta_hidden_norm_ratio_mean": 0.01,
                "delta_hidden_norm_ratio_p95": 0.02,
                "delta_norm_mean": 1.0,
                "hidden_norm_mean": 100.0,
                "value_norm_mean": 0.5,
                "conv_delta_norm_mean": 0.25,
                "groups": {
                    "text_only": {
                        "tokens": 4,
                        "context_gate_mean": 0.25,
                        "context_gate_std": 0.0,
                        "context_gate_min": 0.25,
                        "context_gate_max": 0.25,
                        "context_gate_p95": 0.25,
                        "delta_hidden_norm_ratio_mean": 0.01,
                        "delta_hidden_norm_ratio_p95": 0.01,
                        "delta_norm_mean": 1.0,
                        "hidden_norm_mean": 100.0,
                        "value_norm_mean": 0.5,
                        "conv_delta_norm_mean": 0.25,
                    }
                },
            },
        )
        accumulator.add(
            "layer_1",
            {
                "tokens": 30,
                "context_gate_mean": 0.6,
                "context_gate_std": 0.2,
                "context_gate_min": 0.1,
                "context_gate_max": 0.9,
                "context_gate_p95": 0.8,
                "delta_hidden_norm_ratio_mean": 0.03,
                "delta_hidden_norm_ratio_p95": 0.04,
                "delta_norm_mean": 3.0,
                "hidden_norm_mean": 100.0,
                "value_norm_mean": 1.5,
                "conv_delta_norm_mean": 0.75,
                "groups": {
                    "text_only": {
                        "tokens": 6,
                        "context_gate_mean": 0.75,
                        "context_gate_std": 0.0,
                        "context_gate_min": 0.75,
                        "context_gate_max": 0.75,
                        "context_gate_p95": 0.75,
                        "delta_hidden_norm_ratio_mean": 0.03,
                        "delta_hidden_norm_ratio_p95": 0.03,
                        "delta_norm_mean": 3.0,
                        "hidden_norm_mean": 100.0,
                        "value_norm_mean": 1.5,
                        "conv_delta_norm_mean": 0.75,
                    }
                },
            },
        )

        summary = accumulator.summary()

        self.assertEqual(summary["layer_1"]["num_forwards"], 2)
        self.assertEqual(summary["layer_1"]["tokens"], 40)
        self.assertAlmostEqual(summary["layer_1"]["context_gate_mean"], 0.55)
        self.assertAlmostEqual(summary["layer_1"]["delta_hidden_norm_ratio_mean"], 0.025)
        self.assertAlmostEqual(summary["layer_1"]["groups"]["text_only"]["context_gate_mean"], 0.55)
        self.assertEqual(summary["layer_1"]["groups"]["text_only"]["tokens"], 10)


if __name__ == "__main__":
    unittest.main()
