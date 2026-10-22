"""Four-code SID parsing and Beauty routing regressions."""
import unittest
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from test_engram_structure import config
from onerec_llm.models.memory_adapter import build_memory_adapter
from onerec_llm.models.memory_adapter.structure import build_structure, configure_structure_tokens

RANGES = [[20, 29], [30, 39], [40, 49], [50, 59]]
ITEM = [10, 20, 30, 40, 50, 11]


class AmazonStructureTests(unittest.TestCase):
    def context(self, tokens, samples=None):
        ids = torch.tensor([tokens])
        return build_structure(ids, torch.zeros_like(ids) if samples is None else torch.tensor([samples]),
                               torch.ones_like(ids, dtype=torch.bool), config(adapter_sid_code_ranges=RANGES))

    def test_tokenizer_ranges(self):
        vocab = {'<|sid_begin|>': 10, '<|sid_end|>': 11}
        for letter, (lo, hi) in zip('abcd', RANGES):
            vocab.update({f'<s_{letter}_{i-lo}>': i for i in range(lo, hi+1)})
        c = config()
        configure_structure_tokens(c, SimpleNamespace(get_vocab=lambda: vocab))
        self.assertEqual(c.adapter_sid_code_ranges, RANGES)
        del vocab['<s_b_5>']
        with self.assertRaises(ValueError):
            configure_structure_tokens(c, SimpleNamespace(get_vocab=lambda: vocab))

    def test_saved_beauty_tokenizer_and_config_restore(self):
        import json
        from transformers import AutoTokenizer
        root = Path(__file__).resolve().parents[2]
        tokenizer = AutoTokenizer.from_pretrained(root / 'hf_models/Qwen3-0.6B_itemic', local_files_only=True)
        c = config()
        configure_structure_tokens(c, tokenizer)
        self.assertEqual(c.adapter_sid_code_ranges,
                         [[151669, 151924], [151925, 152180], [152181, 152436], [152437, 152692]])
        ids = torch.tensor([tokenizer.encode('<|sid_begin|><s_a_1><s_b_2><s_c_3><s_d_4><|sid_end|><|sid_begin|>',
                                              add_special_tokens=False)])
        context = build_structure(ids, torch.zeros_like(ids), torch.ones_like(ids, dtype=torch.bool), c)
        self.assertTrue(context['history_valid'][0][0, -1])
        self.assertEqual(context['history_positions'][0][0, -1], 4)
        path = root / 'pretrain/tools/model_converter/convert_checkpoint_to_hf.py'
        spec = importlib.util.spec_from_file_location('amazon_restore', path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / 'step1/global_step1'
            output = checkpoint / 'converted'
            output.mkdir(parents=True)
            (output / 'config.json').write_text('{}')
            trained = {key: value for key, value in c.to_dict().items() if key.startswith('adapter_')}
            (run / 'structure_adapter_config.json').write_text(json.dumps(trained))
            self.assertTrue(converter.restore_structure_config(checkpoint, output))
            self.assertEqual(json.loads((output / 'config.json').read_text())['adapter_sid_code_ranges'],
                             c.adapter_sid_code_ranges)

    def test_complete_partial_malformed_and_boundaries(self):
        c = self.context(ITEM + [10, 21, 31, 41, 11, 10])
        self.assertEqual(c['phase'][0, :6].tolist(), list(range(6)))
        self.assertTrue(c['history_valid'][0][0, -1])
        self.assertFalse(c['history_valid'][1][0, -1])
        self.assertEqual(c['histories'][0][0, -1].tolist(), [20, 30])
        self.assertEqual(c['history_positions'][0][0, -1], 4)
        self.assertFalse(self.context(ITEM[:-1] + [10])['history_valid'][0].any())
        self.assertFalse(self.context(ITEM + [10], [0]*6 + [1])['history_valid'][0][0, -1])
        self.assertFalse(self.context(ITEM + [10], [0]*3 + [1]*4)['history_valid'][0].any())

    def test_both_branches_receive_gradients(self):
        for kind in ('engram_structure', 'lngram_structure'):
            with self.subTest(kind=kind):
                c = config(adapter_type=kind, adapter_sid_code_ranges=RANGES,
                           adapter_itm_order=1, adapter_lngram_bits_per_route=2)
                adapter = build_memory_adapter(c, 16)
                with torch.no_grad():
                    for name, p in adapter.named_parameters():
                        if 'memories' in name or '.tables.' in name:
                            p.normal_(0, .1)
                adapter(torch.randn(1, 7, 16), structure_context=self.context(ITEM+[10])).square().sum().backward()
                for branch in (adapter.spm, adapter.itm):
                    tables = branch.memories if kind == 'engram_structure' else branch.tables.values()
                    self.assertGreater(sum(t.weight.grad.abs().sum().item() for t in tables), 0)

    def test_beauty_training_routes(self):
        from onerec_llm.data.qwen3_dataset import Qwen3ChatCompletionDataset
        ds = Qwen3ChatCompletionDataset.__new__(Qwen3ChatCompletionDataset)
        ds.max_sample_length = 20
        ds._process_completion = lambda sample: {'input_ids': torch.tensor([ITEM])}
        for source, enabled in [('LETTER_Beauty_Sequence_Pretrain', True), ('LETTER_Beauty_sft', True),
                                ('LETTER_Sports_Sequence_Pretrain', True), ('LETTER_Sports_sft', True),
                                ('LETTER_Toys_Sequence_Pretrain', True), ('LETTER_Toys_sft', True),
                                ('LETTER_Yelp_Sequence_Pretrain', True), ('LETTER_Yelp_sft', True),
                                ('LETTER_Instruments_Sequence_Pretrain', True), ('LETTER_Instruments_sft', True),
                                ('LETTER_Instruments_ItemUnderstand_Pretrain', False),
                                ('LETTER_Beauty_ItemUnderstand_Pretrain', False), ('general', False)]:
            out = ds._process({'json': {'segments': []}, 'epoch_idx': 0}, source)
            self.assertEqual(out['itm_sequence_mask'].all().item(), enabled, source)

    def test_remote_export_and_engram_cached_logits(self):
        from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from transformers import AutoModelForCausalLM
        path = Path(__file__).resolve().parents[1] / 'tools/model_converter/convert_checkpoint_to_hf.py'
        spec = importlib.util.spec_from_file_location('amazon_converter', path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        ids = torch.tensor([ITEM + [10, 21, 31, 41, 51, 11, 10]])
        for kind in ('engram_structure', 'lngram_structure'):
            model = Qwen3ForCausalLM(config(adapter_type=kind, adapter_sid_code_ranges=RANGES,
                                          adapter_lngram_bits_per_route=2)).eval()
            with tempfile.TemporaryDirectory() as directory, torch.no_grad():
                model.save_pretrained(directory)
                converter.copy_remote_code_bundle(directory)
                converter.update_remote_code_config(directory)
                loaded = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True).eval()
                self.assertEqual(loaded.config.adapter_sid_code_ranges, RANGES)
                expected = model(ids, itm_sequence_mask=torch.tensor([True]), use_cache=False).logits
                actual = loaded(ids, itm_sequence_mask=torch.tensor([True]), use_cache=False).logits
                torch.testing.assert_close(actual, expected)
                if kind == 'engram_structure':
                    cache = None
                    parts = []
                    for pos in range(ids.shape[1]):
                        out = loaded(ids[:, pos:pos+1], past_key_values=cache, use_cache=True,
                                     itm_sequence_mask=torch.tensor([True]))
                        cache = out.past_key_values
                        parts.append(out.logits)
                    torch.testing.assert_close(torch.cat(parts, 1), expected, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
