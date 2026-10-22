"""Exact ITM artifacts, initialization, and causal inference contracts."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'pretrain'), str(ROOT / 'pretrain/tests'), str(ROOT)]
import torch
from process_data.mine_exact_itm_patterns import mine
from test_engram_structure import config, I1, I2, I3
from onerec_llm.models.memory_adapter import structure
from onerec_llm.models.memory_adapter import build_memory_adapter
from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def exact_config(**kwargs):
    values = dict(adapter_itm_mode='exact', adapter_itm_order=1,
                  adapter_itm_mapping=[0] + [-1] * 99)
    values.update(kwargs)
    return config(**values)


class ExactArtifactTests(unittest.TestCase):
    def test_ab_to_a_exact_and_prior_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            path, data = self.artifact(d)
            data['pattern'] = 'directed_adjacent_to_a'
            path.write_text(json.dumps(data))
            c=config(adapter_itm_mode='exact',adapter_itm_order=1)
            structure.configure_exact_itm(c,path,pattern_min_frequency=5)
            self.assertEqual(c.adapter_itm_mapping,[0]+[-1]*99)
            self.assertEqual(build_memory_adapter(c,16).itm.active_slots,1)
            data['format']='itm_successor_prior_v1'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                structure.configure_exact_itm(config(adapter_itm_mode='exact'),path)

    def artifact(self, directory):
        root = Path(directory)
        inter, index = root / 'x.inter.json', root / 'x.index.json'
        # Item 999 deliberately absent: a test target must never be looked up.
        inter.write_text(json.dumps({str(i): [1, 2, 999] for i in range(5)}))
        index.write_text(json.dumps({'1': ['<s_a_0>', '<s_b_0>'],
                                     '2': ['<s_a_1>', '<s_b_1>']}))
        return mine(inter, index, root / 'patterns', base=10)

    def test_directed_sources_test_exclusion_and_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            _, p = self.artifact(d)
            self.assertEqual(p['active_sources'], [[0, 0]])
            self.assertEqual(p['source_slots'], 1)
            self.assertEqual(p['edge_counts'], [dict(source=[0, 0], target=[1, 1], count=5)])
            self.assertEqual(p['mapping'][11], -1)

    def test_multiple_successors_share_one_slot_and_threshold_is_per_edge(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            index, inter = root / 'index.json', root / 'inter.json'
            index.write_text(json.dumps({str(i): [f'<s_a_{i}>', f'<s_b_{i}>'] for i in (1,2,3)}))
            inter.write_text(json.dumps({str(i): [1, 2 if i < 5 else 3, 999] for i in range(10)}))
            _, p = mine(inter, index, root / 'patterns', base=10)
            self.assertEqual(p['source_slots'], 1)
            self.assertEqual(p['qualifying_edges'], 2)
            with self.assertRaises(ValueError):
                mine(inter, index, root / 'none', min_frequency=6, base=10)

    def test_real_letter_artifacts_load_with_saved_tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ROOT / 'hf_models/Qwen3-0.6B_itemic', local_files_only=True)
        for dataset, count in (('Beauty',481), ('Instruments',256), ('Sports',463), ('Toys',400), ('Yelp',58)):
            c = config(adapter_itm_mode='exact', adapter_itm_order=1)
            structure.configure_structure_tokens(c, tok)
            structure.configure_exact_itm(c, ROOT / f'raw_data/letter/{dataset}/itm_patterns/ab_to_ab_exact_min5.json', tok)
            adapter = build_memory_adapter(c, 16)
            self.assertEqual(adapter.itm.active_slots, count)

    def test_checkpointing_dynamic_gate_and_zero_masks(self):
        torch.manual_seed(10)
        model = Qwen3ForCausalLM(exact_config(adapter_use_spm=False, adapter_fixed_context_gate=None)).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        ids = torch.tensor([I1 + I2 + [10]])
        out = model(ids, sample_idx=torch.zeros_like(ids), itm_sequence_mask=torch.tensor([True]), use_cache=False)
        torch.nn.functional.cross_entropy(out.logits[:,5], ids[:,6]).backward()
        itm = model.model.layers[0].memory_adapter.itm
        self.assertGreater(itm.key_proj.weight.grad.abs().sum().item(), 0)
        c = structure.build_structure(ids, torch.zeros_like(ids), torch.ones_like(ids,dtype=torch.bool), model.config)
        delta = model.model.layers[0].memory_adapter(torch.randn(1,11,16), structure_context=c)
        self.assertEqual(delta[:,10].abs().sum().item(), 0)
        c['itm_write'][:] = False
        self.assertEqual(model.model.layers[0].memory_adapter(torch.randn(1,11,16), structure_context=c).abs().sum().item(), 0)

    def test_loader_embeds_and_rejects_changed_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = self.artifact(d)
            c = config(adapter_itm_mode='exact', adapter_itm_order=1)
            structure.configure_exact_itm(c, path, pattern_min_frequency=5)
            with self.assertRaises(ValueError):
                structure.configure_exact_itm(c, path, pattern_min_frequency=10)
            with self.assertRaises(ValueError):
                structure.configure_exact_itm(c, path.parent)
            self.assertEqual(c.adapter_itm_mapping, [0] + [-1] * 99)
            payload = json.loads(path.read_text())
            payload['mapping'][0], payload['mapping'][11] = -1, 0
            payload['active_sources'] = [[1, 1]]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                structure.configure_exact_itm(c, path)

    def test_meta_construction_and_missing_memory_initialization(self):
        from onerec_llm.training.distributed import initialize_allowed_missing_parameter
        with torch.device('meta'):
            adapter = build_memory_adapter(exact_config(), 16)
        self.assertTrue(all(p.is_meta for p in adapter.parameters()))
        self.assertIsNone(adapter.itm.slot_lookup)
        value = initialize_allowed_missing_parameter('model.layers.0.memory_adapter.itm.memories.0.weight',
                                                     (1, 4), memory_init='zero')
        self.assertEqual(value.abs().sum().item(), 0)

    def test_invalid_mapping_and_mode(self):
        for kwargs in (dict(adapter_itm_mapping=[0]), dict(adapter_itm_order=2),
                       dict(adapter_itm_mapping=[0, 0] + [-1] * 98),
                       dict(adapter_itm_mapping=[-1] * 100), dict(adapter_type='lngram_structure')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_memory_adapter(exact_config(**kwargs), 16)

    def test_resume_restores_identity_and_rejects_mode_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); checkpoint = root / 'step1/global_step1'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'structure_adapter_config.json').write_text(json.dumps({
                'adapter_itm_mode': 'exact', 'adapter_itm_mapping': [0] + [-1] * 99}))
            c = config(adapter_itm_mode='exact', adapter_itm_order=1)
            structure.restore_exact_itm_for_resume(c, checkpoint)
            self.assertEqual(c.adapter_itm_mapping, [0] + [-1] * 99)
            with self.assertRaises(ValueError):
                structure.restore_exact_itm_for_resume(config(), checkpoint)
            with self.assertRaises(ValueError):
                structure.restore_exact_itm_for_resume(exact_config(adapter_itm_mapping=[-1,0]+[-1]*98), checkpoint)

    def test_resume_rejects_changed_split_structure_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            saved = exact_config(adapter_spm_hidden_size=3, adapter_itm_hidden_size=7)
            (Path(d) / 'structure_adapter_config.json').write_text(json.dumps({
                **{k: v for k, v in saved.to_dict().items() if k.startswith('adapter_')},
                'adapter_itm_mapping': saved.adapter_itm_mapping}))
            compatible = exact_config(adapter_spm_hidden_size=3, adapter_itm_hidden_size=7)
            structure.restore_exact_itm_for_resume(compatible, d)
            changed = exact_config(adapter_spm_hidden_size=4, adapter_itm_hidden_size=7)
            with self.assertRaisesRegex(ValueError, 'adapter_spm_hidden_size'):
                structure.restore_exact_itm_for_resume(changed, d)


class ExactInferenceTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('EXACT_ITM_GPU_TEST') == '1', 'opt-in CUDA FSDP test')
    def test_fsdp_meta_load_and_optimizer_step(self):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from onerec_llm.training.distributed import shard_model, load_from_full_model_state_dict
        with tempfile.TemporaryDirectory() as d:
            dist.init_process_group('nccl', init_method='file://' + str(Path(d) / 'rendezvous'), rank=0, world_size=1)
            try:
                torch.cuda.set_device(0)
                mesh = init_device_mesh('cuda', (1,))
                c = exact_config()
                reference = Qwen3ForCausalLM(c)
                base_state = {k:v for k,v in reference.state_dict().items() if '.memory_adapter.' not in k}
                with torch.device('meta'):
                    model = Qwen3ForCausalLM(c)
                shard_model(model, cpu_offload=False, dp_mesh=mesh, fp32_weight=True)
                load_from_full_model_state_dict(model, base_state, memory_init='normal')
                model.model.rotary_emb.inv_freq = reference.model.rotary_emb.inv_freq.cuda()
                for name, p in model.named_parameters():
                    p.requires_grad_('memory_adapter' in name)
                ids = torch.tensor([I1 + I2], device='cuda')
                out = model(ids, sample_idx=torch.zeros_like(ids), itm_sequence_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False)
                loss = torch.nn.functional.cross_entropy(out.logits[:, 5].float(), ids[:, 6])
                loss.backward()
                table = model.model.layers[0].memory_adapter.itm.memories[0].weight
                self.assertGreater(table.grad.to_local().abs().sum().item(), 0)
                before = table.to_local().detach().clone()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
                optimizer.step()
                self.assertFalse(torch.equal(before, table.to_local()))
                self.assertFalse(any('slot_lookup' in k for k in model.state_dict()))
            finally:
                dist.destroy_process_group()

    def test_full_cached_future_and_beam(self):
        torch.manual_seed(17)
        model = Qwen3ForCausalLM(exact_config()).eval()
        ids = torch.tensor([I1 + [7] + I2 + [10]])
        route = torch.tensor([True])
        with torch.no_grad():
            whole = model(ids, use_cache=False, itm_sequence_mask=route).logits
            cache, parts = None, []
            for pos in range(ids.shape[1]):
                out = model(ids[:, pos:pos+1], use_cache=True, past_key_values=cache,
                            itm_sequence_mask=route)
                cache = out.past_key_values
                parts.append(out.logits)
            torch.testing.assert_close(whole, torch.cat(parts, 1), atol=2e-6, rtol=2e-5)
            other = ids.clone(); other[0, 7:] = torch.tensor(I3[1:] + [10])
            changed = model(other, use_cache=False, itm_sequence_mask=route).logits
            torch.testing.assert_close(whole[:, :7], changed[:, :7])
            prompt = torch.tensor([[0, 0] + I1 + [10], [7, 8] + I2 + [10]])
            kw = dict(attention_mask=(prompt != 0).long(), itm_sequence_mask=torch.tensor([True, True]),
                      max_new_tokens=3, num_beams=3, do_sample=False)
            self.assertTrue(torch.equal(model.generate(prompt, use_cache=False, **kw),
                                        model.generate(prompt, use_cache=True, **kw)))

    def test_remote_export_is_self_contained(self):
        from transformers import AutoModelForCausalLM
        path = ROOT / 'pretrain/tools/model_converter/convert_checkpoint_to_hf.py'
        spec = importlib.util.spec_from_file_location('exact_converter', path)
        converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
        model = Qwen3ForCausalLM(exact_config()).eval()
        ids = torch.tensor([I1 + I2 + [10]])
        with tempfile.TemporaryDirectory() as d, torch.no_grad():
            expected = model(ids, use_cache=False, itm_sequence_mask=torch.tensor([True])).logits
            model.save_pretrained(d)
            converter.copy_remote_code_bundle(d); converter.update_remote_code_config(d)
            loaded = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()
            actual = loaded(ids, use_cache=False, itm_sequence_mask=torch.tensor([True])).logits
            torch.testing.assert_close(actual, expected)
            self.assertEqual(loaded.config.adapter_itm_mapping, model.config.adapter_itm_mapping)


if __name__ == '__main__':
    unittest.main()
