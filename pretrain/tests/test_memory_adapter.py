import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onerec_llm.models.memory_adapter import build_memory_adapter
from onerec_llm.models.memory_adapter.engram import EngramMemoryAdapter
from onerec_llm.models.memory_adapter.lngram import LngramMemoryAdapter
from onerec_llm.models.qwen3.configuration_qwen3 import Qwen3Config
from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from onerec_llm.models.trainable_parameters import configure_trainable_parameters


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)
        self.memory_adapter = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)


class TestMemoryAdapterWiring(unittest.TestCase):
    def test_build_memory_adapter_returns_delta_with_same_shape(self):
        config = Qwen3Config(
            adapter_type="engram",
            adapter_hidden_size=8,
            adapter_memory_size=16,
            adapter_ngram_order=2,
            adapter_topk=4,
        )
        adapter = build_memory_adapter(config, hidden_size=4)
        hidden = torch.randn(2, 3, 4)
        delta = adapter(hidden, input_ids=torch.ones(2, 3, dtype=torch.long))
        self.assertEqual(delta.shape, hidden.shape)
        self.assertFalse(hasattr(adapter, "gate"))

    def test_engram_uses_two_and_three_gram_hash_tables(self):
        adapter = EngramMemoryAdapter(
            hidden_size=16,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=3,
            hash_heads=2,
        )
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

        lookup_ids = adapter._make_lookup_ids(
            input_ids=input_ids,
            seq_len=input_ids.shape[1],
            device=input_ids.device,
        )

        self.assertEqual(lookup_ids.shape, (1, 4, 4))
        self.assertEqual(len(adapter.memories), 4)

    def test_engram_hash_depends_on_prefix_context(self):
        adapter = EngramMemoryAdapter(
            hidden_size=16,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=2,
            hash_heads=1,
        )
        input_ids = torch.tensor(
            [
                [10, 99],
                [20, 99],
            ],
            dtype=torch.long,
        )

        lookup_ids = adapter._make_lookup_ids(
            input_ids=input_ids,
            seq_len=input_ids.shape[1],
            device=input_ids.device,
        )

        self.assertNotEqual(lookup_ids[0, 1, 0].item(), lookup_ids[1, 1, 0].item())

    def test_engram_depthwise_conv_is_in_forward_path(self):
        adapter = EngramMemoryAdapter(
            hidden_size=16,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=2,
            hash_heads=1,
        )
        hidden = torch.randn(2, 4, 16)
        input_ids = torch.tensor(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ],
            dtype=torch.long,
        )

        with torch.no_grad():
            delta_before = adapter(hidden, input_ids=input_ids)
            adapter.conv.weight.fill_(0.1)
            delta_after = adapter(hidden, input_ids=input_ids)

        self.assertFalse(torch.allclose(delta_before, delta_after))

    def test_adapter_layers_only_builds_selected_layers(self):
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            adapter_type="engram",
            adapter_layers="1,3",
            adapter_hidden_size=4,
            adapter_memory_size=17,
            adapter_ngram_order=3,
            adapter_topk=2,
        )

        layer0 = Qwen3DecoderLayer(config, layer_idx=0)
        layer1 = Qwen3DecoderLayer(config, layer_idx=1)

        self.assertIsNone(layer0.memory_adapter)
        self.assertIsInstance(layer1.memory_adapter, EngramMemoryAdapter)

    def test_baseline_without_adapter_type_does_not_build_null_adapter(self):
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            adapter_type=None,
            adapter_layers=None,
        )

        layer = Qwen3DecoderLayer(config, layer_idx=0)

        self.assertIsNone(layer.memory_adapter)

    def test_build_memory_adapter_rejects_unknown_type(self):
        config = Qwen3Config(adapter_type="unknown")
        with self.assertRaises(ValueError):
            build_memory_adapter(config, hidden_size=4)

    def test_build_memory_adapter_rejects_unknown_and_supports_lngram(self):
        with self.assertRaises(ValueError):
            build_memory_adapter(Qwen3Config(adapter_type="l3"), hidden_size=4)
        adapter = build_memory_adapter(
            Qwen3Config(adapter_type="lngram", adapter_lngram_bits_per_route=2,
                        adapter_hidden_size=4, adapter_ngram_order=2),
            hidden_size=4,
        )
        self.assertIsInstance(adapter, LngramMemoryAdapter)


class TestAdapterTrainability(unittest.TestCase):
    def test_adapter_mode_only_leaves_adapter_trainable(self):
        model = DummyModel()
        args = SimpleNamespace(
            finetune_mode="adapter",
            adapter_type="engram",
            start_optimize_embedding_index=0,
        )

        configure_trainable_parameters(model, args)

        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.memory_adapter.weight.requires_grad)
        self.assertFalse(model.mlp.gate_proj.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)

    def test_full_mode_keeps_parameters_trainable(self):
        model = DummyModel()
        args = SimpleNamespace(
            finetune_mode="full",
            adapter_type=None,
            start_optimize_embedding_index=0,
        )

        configure_trainable_parameters(model, args)

        self.assertTrue(model.backbone.weight.requires_grad)
        self.assertTrue(model.memory_adapter.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
