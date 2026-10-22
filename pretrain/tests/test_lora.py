import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onerec_llm.models.lora import LoRALinear, apply_lora_to_model, merge_lora_state_dict
from onerec_llm.models.trainable_parameters import configure_trainable_parameters


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.o_proj = torch.nn.Linear(4, 4, bias=False)


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(4, 4, bias=False)


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()
        self.memory_adapter = torch.nn.Linear(4, 4, bias=False)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.other = torch.nn.Linear(4, 4, bias=False)


class TestLoRALinear(unittest.TestCase):
    def test_lora_linear_is_initially_equivalent_to_base_linear(self):
        base = torch.nn.Linear(4, 3, bias=False)
        lora = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
        x = torch.randn(5, 4)

        self.assertTrue(torch.allclose(lora(x), torch.nn.functional.linear(x, base.weight)))

    def test_apply_lora_replaces_target_modules_on_selected_layers(self):
        model = TinyModel()

        replaced = apply_lora_to_model(
            model,
            target_modules="q_proj,v_proj",
            layers="1",
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )

        self.assertEqual(replaced, 2)
        self.assertIsInstance(model.model.layers[1].self_attn.q_proj, LoRALinear)
        self.assertIsInstance(model.model.layers[1].self_attn.v_proj, LoRALinear)
        self.assertNotIsInstance(model.model.layers[0].self_attn.q_proj, LoRALinear)
        self.assertNotIsInstance(model.model.layers[1].mlp.down_proj, LoRALinear)

    def test_merge_lora_state_dict_removes_lora_keys_and_updates_weight(self):
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(3, 4),
            "model.layers.0.self_attn.q_proj.lora_A": torch.ones(2, 4),
            "model.layers.0.self_attn.q_proj.lora_B": torch.ones(3, 2),
        }

        merged = merge_lora_state_dict(state_dict, lora_alpha=4.0, lora_rank=2)

        self.assertNotIn("model.layers.0.self_attn.q_proj.lora_A", merged)
        self.assertNotIn("model.layers.0.self_attn.q_proj.lora_B", merged)
        expected = torch.full((3, 4), 4.0)
        self.assertTrue(torch.allclose(merged["model.layers.0.self_attn.q_proj.weight"], expected))


class TestLoRATrainability(unittest.TestCase):
    def test_sid_only_mode_only_leaves_itemic_embeddings_trainable(self):
        model = TinyModel()
        args = SimpleNamespace(
            finetune_mode="sid_only",
            freeze_llm=False,
            start_optimize_embedding_index=4,
        )

        configure_trainable_parameters(model, args)

        self.assertFalse(model.other.weight.requires_grad)
        self.assertFalse(model.model.layers[0].self_attn.q_proj.weight.requires_grad)
        self.assertFalse(model.model.layers[0].memory_adapter.weight.requires_grad)
        self.assertTrue(model.model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)

    def test_lora_mode_only_leaves_lora_and_itemic_embeddings_trainable(self):
        model = TinyModel()
        apply_lora_to_model(
            model,
            target_modules="q_proj,v_proj",
            layers=None,
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        args = SimpleNamespace(
            finetune_mode="lora",
            freeze_llm=False,
            start_optimize_embedding_index=4,
        )

        configure_trainable_parameters(model, args)

        self.assertFalse(model.other.weight.requires_grad)
        self.assertFalse(model.model.layers[0].self_attn.q_proj.weight.requires_grad)
        self.assertTrue(model.model.layers[0].self_attn.q_proj.lora_A.requires_grad)
        self.assertTrue(model.model.layers[0].self_attn.q_proj.lora_B.requires_grad)
        self.assertTrue(model.model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)

    def test_adapter_lora_mode_leaves_adapter_lora_and_itemic_embeddings_trainable(self):
        model = TinyModel()
        apply_lora_to_model(
            model,
            target_modules="q_proj,v_proj",
            layers=None,
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        args = SimpleNamespace(
            finetune_mode="adapter_lora",
            freeze_llm=False,
            start_optimize_embedding_index=4,
        )

        configure_trainable_parameters(model, args)

        self.assertFalse(model.other.weight.requires_grad)
        self.assertFalse(model.model.layers[0].self_attn.q_proj.weight.requires_grad)
        self.assertTrue(model.model.layers[0].self_attn.q_proj.lora_A.requires_grad)
        self.assertTrue(model.model.layers[0].self_attn.q_proj.lora_B.requires_grad)
        self.assertTrue(model.model.layers[0].memory_adapter.weight.requires_grad)
        self.assertTrue(model.model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)

    def test_adapter_partial_mode_only_unfreezes_selected_layer_modules(self):
        model = TinyModel()
        args = SimpleNamespace(
            finetune_mode="adapter_partial",
            freeze_llm=False,
            start_optimize_embedding_index=4,
            partial_train_layers="1",
            partial_train_target_modules="q_proj,o_proj",
        )

        configure_trainable_parameters(model, args)

        self.assertFalse(model.other.weight.requires_grad)
        self.assertFalse(model.model.layers[0].self_attn.q_proj.weight.requires_grad)
        self.assertTrue(model.model.layers[1].self_attn.q_proj.weight.requires_grad)
        self.assertFalse(model.model.layers[1].self_attn.k_proj.weight.requires_grad)
        self.assertFalse(model.model.layers[1].self_attn.v_proj.weight.requires_grad)
        self.assertTrue(model.model.layers[1].self_attn.o_proj.weight.requires_grad)
        self.assertFalse(model.model.layers[1].mlp.down_proj.weight.requires_grad)
        self.assertTrue(model.model.layers[0].memory_adapter.weight.requires_grad)
        self.assertTrue(model.model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)

    def test_adapter_partial_mode_requires_explicit_layer_selector(self):
        model = TinyModel()
        args = SimpleNamespace(
            finetune_mode="adapter_partial",
            freeze_llm=False,
            start_optimize_embedding_index=0,
            partial_train_layers=None,
            partial_train_target_modules="q_proj,o_proj",
        )

        with self.assertRaises(ValueError):
            configure_trainable_parameters(model, args)


if __name__ == "__main__":
    unittest.main()
