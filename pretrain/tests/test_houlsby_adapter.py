import unittest
from types import SimpleNamespace

import torch
from torch import nn

from onerec_llm.models.houlsby_adapter import HoulsbyAdapter
from onerec_llm.models.trainable_parameters import configure_trainable_parameters


class HoulsbyAdapterTests(unittest.TestCase):
    def test_houlsby_adapter_is_residual_bottleneck(self):
        adapter = HoulsbyAdapter(hidden_size=8, bottleneck_size=4)
        x = torch.randn(2, 3, 8)
        y = adapter(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), (8 * 4 + 4) + (4 * 8 + 8))


    def test_houlsby_mode_trains_only_houlsby_modules_and_itemic_rows(self):
        model = nn.Module()
        model.houlsby_adapter = HoulsbyAdapter(8, 4)
        model.backbone = nn.Linear(8, 8)
        model.embed_tokens = nn.Embedding(16, 8)
        model.lm_head = nn.Linear(8, 16, bias=False)
        args = SimpleNamespace(finetune_mode="houlsby_adapter", start_optimize_embedding_index=10)

        configure_trainable_parameters(model, args)

        self.assertTrue(model.houlsby_adapter.down.weight.requires_grad)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
