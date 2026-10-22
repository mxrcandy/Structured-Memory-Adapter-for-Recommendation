"""Loss normalization and strict export contracts for next-a supervision."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import os
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'pretrain'), str(ROOT)]
from tools.model_converter import convert_checkpoint_to_hf as converter


def distributed_aux_worker(rank, rendezvous):
    import torch.distributed as dist
    from onerec_llm.training.itm_aux import normalize_itm_aux
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        for counts in ((1., 3.), (0., 2.), (0., 0.)):
            x = torch.tensor(3., requires_grad=True)
            count = counts[rank]
            stats = torch.stack([x * count, x.new_tensor(count), x.new_tensor(count)])
            loss, metrics = normalize_itm_aux(stats)
            loss.backward()
            dist.all_reduce(x.grad)
            x.grad.div_(2)
            expected = 1. if sum(counts) else 0.
            torch.testing.assert_close(x.grad, torch.tensor(expected))
            assert loss.item() == 3. * expected
            assert metrics['itm_aux_count'] == sum(counts)
    finally:
        dist.destroy_process_group()


class AuxiliaryTrainingTests(unittest.TestCase):
    def module(self):
        name = 'onerec_llm.training.itm_aux'
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing globally normalized auxiliary loss')
        return importlib.import_module(name)

    def test_local_mean_and_zero_have_valid_gradients(self):
        aux = self.module()
        s = torch.tensor([6., 1., 2.], requires_grad=True)
        loss, metrics = aux.normalize_itm_aux(s)
        self.assertEqual(loss.item(), 3.)
        self.assertEqual(metrics['itm_aux_top1'], .5)
        loss.backward()
        self.assertEqual(s.grad[0].item(), .5)
        z = torch.zeros(3, requires_grad=True)
        loss, metrics = aux.normalize_itm_aux(z)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(metrics['itm_aux_count'], 0.)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_two_rank_unequal_and_zero_counts(self):
        self.module()
        with tempfile.TemporaryDirectory() as d:
            torch.multiprocessing.spawn(distributed_aux_worker, args=(str(Path(d) / 'rendezvous'),), nprocs=2)

    def test_weight_inheritance_disable_and_validation(self):
        module = self.module()
        cfg = SimpleNamespace(adapter_type='engram_structure', adapter_use_itm=True,
                              adapter_itm_mode='exact', adapter_itm_aux_loss_weight=.1)
        module.configure_itm_aux_training(cfg)
        self.assertEqual(cfg.adapter_itm_aux_loss_weight, .1)
        self.assertTrue(cfg.adapter_itm_aux_head)
        module.configure_itm_aux_training(cfg, 0)
        self.assertTrue(cfg.adapter_itm_aux_head)
        self.assertEqual(cfg.adapter_itm_aux_loss_weight, 0)
        for invalid in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                module.configure_itm_aux_training(cfg, invalid)
        cfg.adapter_itm_mode = 'default'
        with self.assertRaises(ValueError):
            module.configure_itm_aux_training(cfg, .1)

    def test_chunked_aux_backward_matches_joint_loss(self):
        from onerec_llm.losses import CrossEntropyLoss, ChunkedLossComputer
        import copy
        for zero in (False, True):
            base = torch.nn.Linear(4, 4)
            head = torch.nn.Linear(4, 8)
            base2, head2 = copy.deepcopy(base), copy.deepcopy(head)
            x = torch.randn(1, 5, 4)
            labels = torch.full((1,5), -100) if zero else torch.tensor([[1,2,3,4,-100]])
            ce = CrossEntropyLoss(return_token_loss=True, shift_labels=False)
            hidden = base(x)
            aux = base.weight.square().mean() * .1
            lm, _ = ce(head(hidden), labels)
            (lm + aux).backward()
            computer = ChunkedLossComputer(head2, ce, minibatch_size=2, shift_labels=False)
            result, per_token = computer.forward_and_backward(base2(x), labels,
                additional_loss=.1 * base2.weight.square().mean())
            self.assertAlmostEqual(result.item(), lm.item(), places=5)
            self.assertIsNotNone(per_token)
            for actual, expected in zip(list(base2.parameters())+list(head2.parameters()),
                                        list(base.parameters())+list(head.parameters())):
                torch.testing.assert_close(actual.grad, expected.grad)

    def test_export_rejects_missing_or_orphaned_heads(self):
        self.assertTrue(hasattr(converter, 'validate_itm_aux_export'), 'missing strict auxiliary export validation')
        cfg = dict(adapter_type='engram_structure', adapter_use_itm=True,
                   adapter_itm_mode='exact', adapter_itm_aux_loss_weight=.1,
                   adapter_itm_aux_head=True, adapter_layers='0,1',
                   num_hidden_layers=2, hidden_size=16, adapter_hidden_size=4,
                   adapter_topk=2, adapter_sid_code_ranges=[[20,29],[30,39]])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'config.json').write_text(json.dumps(cfg))
            names = {f'model.layers.{i}.memory_adapter.itm.next_a_head.weight': torch.zeros(10, 8) for i in range(2)}
            from safetensors.torch import save_file
            save_file(names, p / 'model.safetensors')
            converter.validate_itm_aux_export(p)
            names.pop(next(iter(names)))
            save_file(names, p / 'model.safetensors')
            with self.assertRaisesRegex(ValueError, 'head'):
                converter.validate_itm_aux_export(p)

            cfg['adapter_itm_aux_head'] = False
            cfg['adapter_itm_aux_loss_weight'] = 0.
            (p / 'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'head'):
                converter.validate_itm_aux_export(p)

    @unittest.skipUnless(os.environ.get('EXACT_ITM_GPU_TEST') == '1', 'opt-in CUDA FSDP test')
    def test_fsdp_checkpointing_strict_sft_load_and_real_trainer_step(self):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from onerec_llm.training.distributed import shard_model, load_from_full_model_state_dict
        from onerec_llm.training.activations import set_activation_checkpointing
        from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from recipes.train_qwen3 import compute_forward_backward
        from onerec_llm.losses import CrossEntropyLoss, ChunkedLossComputer
        sys.path.insert(0, str(ROOT / 'pretrain/tests'))
        from test_exact_itm import exact_config, I1, I2
        with tempfile.TemporaryDirectory() as d:
            torch.cuda.set_device(0)
            dist.init_process_group('nccl', init_method='file://' + str(Path(d) / 'rendezvous'), rank=0, world_size=1)
            try:
                cfg = exact_config(adapter_itm_aux_loss_weight=.1)
                cfg.use_cache = False
                reference = Qwen3ForCausalLM(cfg)
                full = reference.state_dict()
                with torch.device('meta'):
                    model = Qwen3ForCausalLM(cfg)
                set_activation_checkpointing(model, auto_wrap_policy=Qwen3ForCausalLM.wrap_modules)
                shard_model(model, cpu_offload=False, dp_mesh=init_device_mesh('cuda', (1,)), fp32_weight=True)
                missing = {k: v for k, v in full.items() if '.next_a_head.' not in k}
                with self.assertRaisesRegex(ValueError, 'missing saved next-a head'):
                    load_from_full_model_state_dict(model, missing, require_itm_aux_head=True)
                load_from_full_model_state_dict(model, full, require_itm_aux_head=True)
                model.model.rotary_emb.inv_freq = reference.model.rotary_emb.inv_freq.cuda()
                for name, p in model.named_parameters():
                    p.requires_grad_('memory_adapter' in name)
                ids = torch.tensor([I1+I2], device='cuda')
                batch = dict(input_ids=ids, sample_idx=torch.zeros_like(ids),
                             itm_sequence_mask=torch.tensor([True], device='cuda'), loss_mask=torch.ones_like(ids))
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
                ce = CrossEntropyLoss(return_token_loss=True, shift_labels=False)
                head = model.model.layers[0].memory_adapter.itm.next_a_head.weight
                for chunked in (False, True):
                    optimizer.zero_grad(set_to_none=True)
                    model.config.chunked_loss_computer = chunked
                    model.chunked_loss_computer = chunked
                    compute = ChunkedLossComputer(model.lm_head, ce, minibatch_size=3, shift_labels=False).forward_and_backward if chunked else ce
                    loss, per_token, metrics = compute_forward_backward(model, batch, compute, ce,
                        SimpleNamespace(start_optimize_embedding_index=0, max_grad_norm=0,
                                        use_chunked_loss_computer=chunked), None, optimizer)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertEqual(metrics['itm_aux_count'], 1)
                    self.assertAlmostEqual(metrics['total_loss'], metrics['lm_loss'] + .1 * metrics['itm_aux_loss'])
                    self.assertGreater(head.grad.to_local().abs().sum().item(), 0)
                    before = head.to_local().detach().clone()
                    optimizer.step()
                    self.assertFalse(torch.equal(before, head.to_local()))
            finally:
                dist.destroy_process_group()

    def test_stage2_conversion_remote_reload_and_sft_export(self):
        sys.path.insert(0, str(ROOT / 'pretrain/tests'))
        from test_exact_itm import exact_config, I1, I2
        from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from transformers import AutoModelForCausalLM, AutoModel
        model = Qwen3ForCausalLM(exact_config(adapter_itm_aux_loss_weight=.1))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'base'
            source.mkdir()
            # Deliberately stale base metadata must not override the training run.
            cfg = model.config.to_dict()
            cfg.update(adapter_itm_aux_loss_weight=0., adapter_itm_aux_head=False)
            (source / 'config.json').write_text(json.dumps(cfg))
            for stage in ('stage2', 'sft'):
                run = root / stage
                run.mkdir()
                checkpoint = run / ('dcp' if stage == 'stage2' else 'model.pth')
                if stage == 'stage2':
                    import torch.distributed.checkpoint as dcp
                    dcp.save({'app': {'model': model.state_dict()}}, checkpoint_id=checkpoint)
                else:
                    torch.save({'model': model.state_dict()}, checkpoint)
                saved = {k: v for k, v in model.config.to_dict().items() if k.startswith('adapter_')}
                (run / 'structure_adapter_config.json').write_text(json.dumps(saved))
                output = run / 'export'
                result = subprocess.run([sys.executable, str(Path(converter.__file__)),
                    '--checkpoint_dir', str(checkpoint), '--output_dir', str(output),
                    '--source_hf_model_path', str(source), '--bundle_remote_code', '--dtype', 'bf16'],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                restored, info = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True,
                    local_files_only=True, output_loading_info=True, attn_implementation='eager')
                self.assertEqual(info['missing_keys'], [])
                self.assertEqual(info['unexpected_keys'], [])
                self.assertEqual(restored.config.adapter_itm_mapping, model.config.adapter_itm_mapping)
                self.assertEqual(restored.config.adapter_itm_aux_loss_weight, .1)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value.to(torch.bfloat16).to(restored.state_dict()[name].dtype))
                # The vLLM Transformers backend constructs AutoModel, whose decoder
                # must own and load all auxiliary weights without running the head.
                decoder = AutoModel.from_config(restored.config, trust_remote_code=True).eval()
                decoder.load_state_dict(restored.model.state_dict(), strict=True)
                ids = torch.tensor([I1+I2])
                out = decoder(ids, itm_sequence_mask=torch.tensor([True]), use_cache=False)
                self.assertIsNone(out.itm_aux_stats)
                # Simulate normal SFT forward/update before its own conversion.
                restored.train()
                out = restored(ids, sample_idx=torch.zeros_like(ids), itm_sequence_mask=torch.tensor([True]),
                    itm_aux_loss_mask=torch.ones_like(ids), use_cache=False)
                loss = out.logits.float().square().mean() + .1 * out.itm_aux_stats[0]
                loss.backward()
                optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
                optimizer.step()
                model, source = restored, output


if __name__ == '__main__':
    unittest.main()
