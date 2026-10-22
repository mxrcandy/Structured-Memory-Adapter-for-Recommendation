import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onerec_llm.models.memory_adapter.engram import EngramMemoryAdapter


class TestEngramControls(unittest.TestCase):
    def test_alpha_layer_is_optional_and_scales_final_delta(self):
        base = EngramMemoryAdapter(
            hidden_size=8,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=2,
            hash_heads=1,
        )
        scaled = EngramMemoryAdapter(
            hidden_size=8,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=2,
            hash_heads=1,
            use_alpha_layer=True,
            alpha_init=0.25,
        )
        scaled.load_state_dict(base.state_dict(), strict=False)
        hidden = torch.randn(1, 4, 8)
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

        self.assertIsNone(base.alpha_layer)
        self.assertEqual(tuple(scaled.alpha_layer.shape), (1,))
        self.assertAlmostEqual(scaled.alpha_layer.item(), 0.25)
        self.assertTrue(torch.allclose(
            scaled(hidden, input_ids=input_ids),
            0.25 * base(hidden, input_ids=input_ids),
        ))

    def test_fixed_context_gate_zero_disables_all_residual_paths(self):
        adapter = EngramMemoryAdapter(
            hidden_size=8,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=2,
            hash_heads=1,
            fixed_context_gate=0.0,
        )
        with torch.no_grad():
            adapter.conv.weight.fill_(0.1)

        delta = adapter(
            torch.randn(1, 4, 8),
            input_ids=torch.tensor([[10, 20, 30, 40]], dtype=torch.long),
        )

        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_disabling_conv_returns_only_gated_values(self):
        adapter = EngramMemoryAdapter(
            hidden_size=8,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=2,
            hash_heads=1,
            use_conv=False,
        )
        hidden = torch.randn(1, 4, 8)
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

        with torch.no_grad():
            lookup_ids = adapter._make_lookup_ids(input_ids, input_ids.shape[1], hidden.device)
            memory = adapter._lookup_memory(lookup_ids)
            values = adapter.value_proj(memory)
            query = adapter.query_norm(hidden)
            key = adapter.key_norm(adapter.key_proj(memory))
            gate = torch.sigmoid((query * key).sum(dim=-1, keepdim=True) / (adapter.hidden_size ** 0.5))

        self.assertTrue(torch.allclose(adapter(hidden, input_ids=input_ids), gate * values))

    def test_sid_span_mask_zeros_non_itemic_tokens(self):
        adapter = EngramMemoryAdapter(
            hidden_size=8,
            adapter_hidden_size=4,
            memory_size=17,
            ngram_order=2,
            hash_heads=1,
            use_sid_span=True,
            sid_token_start=100,
            sid_token_end=104,
        )
        delta = adapter(
            torch.randn(1, 5, 8),
            input_ids=torch.tensor([[10, 100, 101, 104, 20]], dtype=torch.long),
        )

        self.assertTrue(torch.equal(delta[:, [0, 4]], torch.zeros_like(delta[:, [0, 4]])))
        self.assertGreater(delta[:, 1:4].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
