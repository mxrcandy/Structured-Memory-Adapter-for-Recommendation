"""CPU numerical regression tests; no vLLM installation or trained weights needed."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pretrain'))
from onerec_llm.models.memory_adapter.lngram_structure import StructuredLngramMemoryAdapter
from onerec_llm.models.memory_adapter.lngram import LngramMemoryAdapter
from onerec_llm.models.memory_adapter.structure import build_structure

spec = importlib.util.spec_from_file_location('symbol_cache', ROOT / 'benchmarks/benchmark/vllm_lngram_cache.py')
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)

# Exercise the real bridge functions without importing CUDA-only vLLM on CPU.
path = ROOT / 'benchmarks/benchmark/vllm_structure.py'
functions = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in
             ('context_to_kwargs', 'kwargs_to_context', 'scheduler_context')]
scope = dict(torch=torch, build_structure=build_structure)
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)


def config(**kwargs):
    values = dict(adapter_use_spm=True, adapter_use_itm=True, adapter_ngram_order=3,
                  adapter_itm_order=2, adapter_lngram_bits_per_route=2,
                  adapter_hidden_size=3, adapter_fixed_context_gate=None,
                  adapter_sid_begin_id=10, adapter_sid_end_id=11,
                  adapter_sid_code_ranges=[[20, 29], [30, 39], [40, 49]])
    values.update(kwargs)
    return SimpleNamespace(**values)


class SymbolCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.c = config()
        self.reference = StructuredLngramMemoryAdapter(self.c, 8).eval()
        with torch.no_grad():
            for name, param in self.reference.named_parameters():
                if '.tables.' in name:
                    param.normal_(0, .4)
        self.cached = copy.deepcopy(self.reference)
        self.caches = cache_module.install_symbol_caches(self.cached)
        for cache in self.caches:
            cache.allocate(32, 8, 4, 'cpu')
        self.ids = [10, 21, 32, 43, 11, 8, 10, 22, 33, 44, 11, 10, 23, 34, 45, 11, 10]
        self.hidden = torch.randn(1, len(self.ids), 8)

    def context(self, sequences, lengths, queries, tables, route=True):
        kwargs = scope['scheduler_context'](sequences, lengths, queries, self.c, route, 'cpu',
                                             block_tables=tables, block_size=4)
        return scope['kwargs_to_context']({k: v[None] for k, v in kwargs.items()}, self.c)

    def full(self, ids, hidden, route=True):
        ids = torch.tensor([ids])
        c = build_structure(ids, torch.zeros_like(ids), torch.full_like(ids, route, dtype=torch.bool), self.c)
        return self.reference(hidden, structure_context=c)

    @torch.inference_mode()
    def test_chunked_prefill_and_each_decode_position(self):
        expected = self.full(self.ids, self.hidden)
        tables = [[7, 3, 11, 2, 8]]
        for chunks in ([len(self.ids)], [1] * len(self.ids), [3, 4, 2, 8]):
            with self.subTest(chunks=chunks):
                for cache in self.caches:
                    cache.gpu_valid.zero_()
                start = 0
                for query in chunks:
                    end = start + query
                    c = self.context([self.ids], [end], [query], tables)
                    got = self.cached(self.hidden[:, start:end], structure_context=c)
                    torch.testing.assert_close(got, expected[:, start:end], rtol=1e-5, atol=1e-6)
                    start = end

    @torch.inference_mode()
    def test_mixed_requests_reordering_and_route(self):
        other = [7, 10, 24, 35, 46, 11, 10, 25, 36, 47, 11, 10, 26, 37, 48, 11, 10]
        hidden2 = torch.randn_like(self.hidden)
        tables = [[0, 1, 2, 3, 4], [8, 9, 10, 11, 12]]
        for route in (False, True, False):
            expected = [self.full(self.ids, self.hidden, route), self.full(other, hidden2, route)]
            start = 0
            for query in (4, 3, 1, 9):
                end = start + query
                order = [0, 1] if start == 0 else [1, 0]
                sequences = [self.ids, other]
                hidden = [self.hidden, hidden2]
                c = self.context([sequences[i] for i in order], [end] * 2, [query] * 2,
                                 [tables[i] for i in order], route)
                got = self.cached(torch.cat([hidden[i][:, start:end] for i in order], 1), structure_context=c)
                want = torch.cat([expected[i][:, start:end] for i in order], 1)
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
                start = end

    @torch.inference_mode()
    def test_itm_order_one_and_single_branch_variants(self):
        for spm, itm in ((True, True), (True, False), (False, True)):
            with self.subTest(spm=spm, itm=itm):
                self.c = config(adapter_use_spm=spm, adapter_use_itm=itm, adapter_itm_order=1,
                                adapter_lngram_bits_per_route=4)
                self.reference = StructuredLngramMemoryAdapter(self.c, 8).eval()
                for name, param in self.reference.named_parameters():
                    if '.tables.' in name:
                        param.normal_(0, .4)
                self.cached = copy.deepcopy(self.reference)
                self.caches = cache_module.install_symbol_caches(self.cached)
                for cache in self.caches:
                    cache.allocate(32, 0, 4, 'cpu')
                expected = self.full(self.ids, self.hidden)
                for pos in range(len(self.ids)):
                    c = self.context([self.ids], [pos + 1], [1], [[7, 3, 11, 2, 8]])
                    got = self.cached(self.hidden[:, pos:pos + 1], structure_context=c)
                    torch.testing.assert_close(got, expected[:, pos:pos + 1], rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_prefix_reuse_and_beam_copy_on_write(self):
        # Share two full blocks, clone the partial block, then diverge.
        prefix = self.ids[:10]
        self.cached(self.hidden[:, :10], structure_context=self.context([prefix], [10], [10], [[0, 1, 2]]))
        for cache in self.caches:
            cache.copy(torch.tensor([[2, 5]]))
        sequences = [prefix + [11, 10, 23], prefix + [11, 10, 24]]
        hidden = [torch.cat([self.hidden[:, :10], torch.randn(1, 3, 8)], 1) for _ in range(2)]
        got = self.cached(torch.cat([x[:, 10:] for x in hidden], 1), structure_context=self.context(
            sequences, [13, 13], [3, 3], [[0, 1, 2, 3], [0, 1, 5, 6]]))
        expected = torch.cat([self.full(ids, h)[:, 10:] for ids, h in zip(sequences, hidden)], 1)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_swap_and_reused_blocks(self):
        self.cached(self.hidden[:, :8], structure_context=self.context([self.ids], [8], [8], [[0, 1]]))
        for cache in self.caches:
            cache.swap(torch.tensor([[0, 0], [1, 1]]), swap_in=False)
            cache.gpu[:2].zero_()
            cache.gpu_valid[:2].zero_()
            cache.swap(torch.tensor([[0, 6], [1, 7]]), swap_in=True)
        got = self.cached(self.hidden[:, 8:], structure_context=self.context(
            [self.ids], [17], [9], [[6, 7, 8, 9, 10]]))
        torch.testing.assert_close(got, self.full(self.ids, self.hidden)[:, 8:], rtol=1e-5, atol=1e-6)
        fresh = torch.randn_like(self.hidden)
        got = self.cached(fresh, structure_context=self.context([self.ids], [17], [17], [[6, 7, 8, 9, 10]]))
        torch.testing.assert_close(got, self.full(self.ids, fresh), rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_missing_history_fails_instead_of_clamping(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing Lngram history'):
            self.cached(self.hidden[:, 6:7], structure_context=self.context([self.ids], [7], [1], [[0, 1]]))

    def test_transport_preserves_history_positions(self):
        ids = torch.tensor([self.ids])
        c = build_structure(ids, torch.zeros_like(ids), torch.ones_like(ids, dtype=torch.bool), self.c)
        restored = scope['kwargs_to_context'](scope['context_to_kwargs'](c), self.c)
        self.assertEqual(set(c), set(restored))
        for actual, expected in zip(restored['history_positions'], c['history_positions']):
            torch.testing.assert_close(actual, expected)

    def test_cache_does_not_change_state_dict_and_budget_matches(self):
        self.assertEqual(set(self.cached.state_dict()), set(self.reference.state_dict()))
        for name, param in self.reference.state_dict().items():
            torch.testing.assert_close(self.cached.state_dict()[name], param, rtol=0, atol=0)
        for cache in self.caches:
            actual = cache.gpu.numel() * cache.gpu.element_size() + cache.gpu_valid.numel()
            self.assertEqual(cache.block_bytes(4) * 32, actual)

    @torch.inference_mode()
    def test_ordinary_lngram_uses_complete_history_slots(self):
        c = config(adapter_type='lngram', adapter_use_spm=False, adapter_use_itm=False)
        # The scheduler must build ordinary token windows, not SID-only windows.
        self.c = c
        reference = LngramMemoryAdapter(c, 8).eval()
        cached = copy.deepcopy(reference)
        caches = cache_module.install_symbol_caches(cached)
        for cache in caches:
            cache.allocate(32, 0, 4, 'cpu')
        expected = reference(self.hidden)
        start = 0
        for query in (3, 4, 2, 8):
            end = start + query
            context = self.context([self.ids], [end], [query], [[7, 3, 11, 2, 8]])
            got = cached(self.hidden[:, start:end], structure_context=context)
            torch.testing.assert_close(got, expected[:, start:end], rtol=1e-5, atol=1e-6)
            start = end

    @torch.inference_mode()
    def test_two_layer_qwen_incremental_hidden_states_match_full_prefix(self):
        from onerec_llm.models.qwen3.configuration_qwen3 import Qwen3Config
        from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3Model

        c = Qwen3Config(vocab_size=80, hidden_size=8, intermediate_size=16,
                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                       head_dim=4, adapter_type='lngram_structure', **vars(self.c))
        c._attn_implementation = 'eager'
        reference = Qwen3Model(c).eval()
        for name, param in reference.named_parameters():
            if '.tables.' in name:
                param.normal_(0, .4)
        incremental = copy.deepcopy(reference)
        caches = cache_module.install_symbol_caches(incremental)
        for cache in caches:
            cache.allocate(16, 0, 4, 'cpu')
        context = None

        def replace_context(module, args, kwargs):
            kwargs['structure_context'] = context
            return args, kwargs

        for layer in incremental.layers:
            layer.memory_adapter.register_forward_pre_hook(replace_context, with_kwargs=True)
        ids = torch.tensor([self.ids])
        expected = reference(ids, itm_sequence_mask=torch.tensor([True]), use_cache=False).last_hidden_state
        kv = None
        start = 0
        for query in (3, 4, 1, 1, 1, 1, 1, 2, 3):
            end = start + query
            context = self.context([self.ids], [end], [query], [[4, 1, 7, 2, 8]])
            output = incremental(ids[:, start:end], past_key_values=kv, use_cache=True,
                                 itm_sequence_mask=torch.tensor([True]))
            kv = output.past_key_values
            torch.testing.assert_close(output.last_hidden_state, expected[:, start:end], rtol=1e-5, atol=1e-6)
            start = end


class FourCodeSymbolCacheTests(SymbolCacheTests):
    def setUp(self):
        super().setUp()
        self.c.adapter_sid_code_ranges.append([50, 59])
        # Retain 17 positions so all existing cache/chunk/block tests also run here.
        self.ids = [10, 21, 32, 43, 54, 11, 10, 22, 33, 44, 55, 11, 10, 23, 34, 45, 56]


if __name__ == '__main__':
    unittest.main()
