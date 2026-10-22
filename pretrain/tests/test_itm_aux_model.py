"""Exact source-memory next-a supervision and explicit forward outputs."""
import unittest
import json
import tempfile
from pathlib import Path
import torch
from test_exact_itm import exact_config, I1, I2, I3
from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from onerec_llm.models.memory_adapter.structure import restore_exact_itm_for_resume


class ITMAuxModelTests(unittest.TestCase):
    def model(self, **kw):
        return Qwen3ForCausalLM(exact_config(adapter_itm_aux_loss_weight=.2,
                                            adapter_use_spm=False, **kw)).train()

    def forward(self, model, ids, mask=None, **kw):
        ids = torch.tensor([ids])
        return model(ids, sample_idx=torch.zeros_like(ids), itm_sequence_mask=torch.tensor([True]),
                     itm_aux_loss_mask=torch.ones_like(ids) if mask is None else torch.tensor([mask]),
                     use_cache=False, **kw)

    def test_head_uses_raw_memory_and_supervises_every_successor(self):
        model = self.model()
        ids = I1 + I2 + I1 + I3
        out = self.forward(model, ids)
        itm = model.model.layers[0].memory_adapter.itm
        self.assertEqual(itm.next_a_head.weight.shape, (10, 8))
        self.assertEqual(out.itm_aux_stats.shape, (3,))
        self.assertEqual(out.itm_aux_stats[2].item(), 2)
        losses = []
        for layer in model.model.layers:
            branch = layer.memory_adapter.itm
            memory = torch.cat([table.weight[0] for table in branch.memories])
            logits = branch.next_a_head(memory).float().expand(2, -1)
            losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor([1,2]), reduction='sum'))
        torch.testing.assert_close(out.itm_aux_stats[0], torch.stack(losses).mean())
        out.itm_aux_stats[0].backward()
        self.assertGreater(itm.next_a_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(itm.memories[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(itm.value_proj.weight.grad)
        self.assertIsNone(model.model.embed_tokens.weight.grad)

    def test_mask_current_position_not_shifted_and_zero_is_differentiable(self):
        model = self.model()
        mask = [0] * 10
        mask[6] = 1
        out = self.forward(model, I1 + I2, mask)
        self.assertEqual(out.itm_aux_stats[2].item(), 0)
        self.assertEqual(out.itm_aux_stats[0].item(), 0)
        out.itm_aux_stats[0].backward()
        mask[5] = 2
        self.assertEqual(self.forward(model, I1 + I2, mask).itm_aux_stats[2].item(), 0)
        mask[5] = 1
        self.assertEqual(self.forward(model, I1 + I2, mask).itm_aux_stats[2].item(), 1)

    def test_sample_boundary_invalid_a_route_and_padding_excluded(self):
        model = self.model()
        ids = torch.tensor([I1 + I2])
        common = dict(itm_aux_loss_mask=torch.ones_like(ids), use_cache=False)
        for samples, route, tokens, attention in [
            ([0]*6+[1]*4, True, ids, None),
            ([0]*10, False, ids, None),
            ([0]*10, True, torch.tensor([I1+[10,7,31,41,11]]), None),
            ([0]*10, True, ids, torch.tensor([[1]*6+[0]*4])),
        ]:
            out = model(tokens, sample_idx=torch.tensor([samples]), itm_sequence_mask=torch.tensor([route]),
                        attention_mask=attention, **common)
            self.assertEqual(out.itm_aux_stats[2].item(), 0)

    def test_checkpointing_and_attention_tuple_keep_stats(self):
        for reentrant in (False, True):
            model = self.model()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': reentrant})
            out = self.forward(model, I1+I2, output_attentions=True)
            self.assertEqual(len(out.attentions), 2)
            out.itm_aux_stats[0].backward()
            self.assertIsNotNone(model.model.layers[0].memory_adapter.itm.next_a_head.weight.grad)

    def test_training_requires_explicit_mask_inference_skips_aux_head_retained(self):
        model = self.model()
        ids = torch.tensor([I1 + I2])
        with self.assertRaisesRegex(ValueError, 'itm_aux_loss_mask'):
            model(ids, sample_idx=torch.zeros_like(ids), itm_sequence_mask=torch.tensor([True]), use_cache=False)
        model.eval()
        self.assertIsNone(model(ids, itm_sequence_mask=torch.tensor([True]), use_cache=False).itm_aux_stats)
        model.config.adapter_itm_aux_loss_weight = 0
        restored = Qwen3ForCausalLM(type(model.config).from_dict(model.config.to_dict())).train()
        self.assertIsNotNone(restored.model.layers[0].memory_adapter.itm.next_a_head)
        self.assertIsNone(restored(ids, sample_idx=torch.zeros_like(ids), itm_sequence_mask=torch.tensor([True]), use_cache=False).itm_aux_stats)

    def test_resume_restores_head_and_saved_weight_over_stale_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = dict(adapter_itm_mode='exact', adapter_itm_mapping=[0]+[-1]*99,
                         adapter_itm_aux_loss_weight=0, adapter_itm_aux_head=True)
            path = Path(tmp) / 'structure_adapter_config.json'
            path.write_text(json.dumps(saved))
            cfg = exact_config(adapter_itm_aux_loss_weight=.7)
            restore_exact_itm_for_resume(cfg, tmp)
            self.assertEqual(cfg.adapter_itm_aux_loss_weight, 0)
            self.assertTrue(cfg.adapter_itm_aux_head)
            del saved['adapter_itm_aux_loss_weight']
            del saved['adapter_itm_aux_head']
            path.write_text(json.dumps(saved))
            restore_exact_itm_for_resume(cfg, tmp)
            self.assertFalse(cfg.adapter_itm_aux_head)
            saved.update(adapter_itm_aux_loss_weight=.2, adapter_itm_aux_head=False)
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                restore_exact_itm_for_resume(cfg, tmp)

    def test_resume_rejects_changed_aux_head_architecture(self):
        saved = exact_config(adapter_itm_aux_loss_weight=.2).to_dict()
        changes = [dict(adapter_layers='0'), dict(adapter_topk=3),
                   dict(adapter_hidden_size=8), dict(adapter_itm_order=2),
                   dict(adapter_itm_hidden_size=12),
                   dict(adapter_sid_code_ranges=[[21,30],[30,39],[40,49]]),
                   dict(adapter_sid_begin_id=12), dict(adapter_sid_end_id=13),
                   dict(adapter_type='lngram_structure'), dict(adapter_use_itm=False),
                   dict(adapter_itm_mode='default')]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'structure_adapter_config.json').write_text(json.dumps(saved))
            for change in changes:
                cfg = exact_config()
                for key, value in change.items():
                    setattr(cfg, key, value)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    restore_exact_itm_for_resume(cfg, tmp)

    def test_resume_normalizes_layer_selection_and_effective_memory_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'structure_adapter_config.json'
            for selection in (None, 'all', '', [0, 1], '1,0'):
                saved = exact_config(adapter_itm_aux_loss_weight=.2,
                                     adapter_layers=selection, adapter_hidden_size=None).to_dict()
                path.write_text(json.dumps(saved))
                cfg = exact_config(adapter_layers=[1,0], adapter_hidden_size=16)
                restore_exact_itm_for_resume(cfg, tmp)
                self.assertTrue(cfg.adapter_itm_aux_head)

    def test_positive_weight_requires_an_adapter_layer(self):
        with self.assertRaisesRegex(ValueError, 'layer'):
            self.model(adapter_layers='99')

    def test_hf_roundtrip_and_legacy_tuple_layout(self):
        model = self.model()
        ids = torch.tensor([I1+I2])
        model.eval()
        old = model(ids, itm_sequence_mask=torch.tensor([True]), use_cache=False,
                    output_attentions=True, return_dict=False)
        self.assertEqual(len(old), 2)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored = Qwen3ForCausalLM.from_pretrained(tmp).train()
            out = self.forward(restored, I1+I2, output_attentions=True, return_dict=False)
            self.assertEqual(len(out), 3)
            self.assertEqual(out[-1][2].item(), 1)
            torch.testing.assert_close(model.model.layers[0].memory_adapter.itm.next_a_head.weight,
                                       restored.model.layers[0].memory_adapter.itm.next_a_head.weight)

    def test_aux_export_uses_itm_specific_hidden_size(self):
        from tools.model_converter import convert_checkpoint_to_hf as converter
        model = self.model(adapter_hidden_size=4, adapter_itm_hidden_size=6)
        itm = model.model.layers[0].memory_adapter.itm
        self.assertEqual(tuple(itm.next_a_head.weight.shape), (10, 12))
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            converter.validate_itm_aux_export(tmp)
            exported = json.loads((Path(tmp) / 'config.json').read_text())
            self.assertEqual(exported['adapter_hidden_size'], 4)
            self.assertEqual(exported['adapter_itm_hidden_size'], 6)

    def test_invalid_configuration(self):
        for kw in [dict(adapter_itm_aux_loss_weight=-1), dict(adapter_itm_aux_loss_weight=float('nan')),
                   dict(adapter_itm_aux_loss_weight=float('inf')), dict(adapter_itm_mode='default'),
                   dict(adapter_use_itm=False), dict(adapter_type=None)]:
            values = dict(adapter_itm_aux_loss_weight=.2)
            values.update(kw)
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                exact_config(**values)

if __name__ == '__main__':
    unittest.main()
