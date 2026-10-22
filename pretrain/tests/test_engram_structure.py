import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onerec_llm.models.memory_adapter import build_memory_adapter
from onerec_llm.models.memory_adapter.structure import build_structure, prepare_structure, StructuredDynamicCache
from onerec_llm.models.qwen3.configuration_qwen3 import Qwen3Config
from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def config(**kw):
    args = dict(vocab_size=80, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                adapter_type='engram_structure', adapter_hidden_size=4,
                adapter_memory_size=31, adapter_topk=2, adapter_ngram_order=3,
                adapter_fixed_context_gate=.5, adapter_use_conv=False,
                adapter_sid_begin_id=10, adapter_sid_end_id=11,
                adapter_sid_code_ranges=[[20,29],[30,39],[40,49]],
                adapter_itm_order=2, pad_token_id=0, eos_token_id=2)
    args.update(kw)
    c=Qwen3Config(**args)
    c._attn_implementation='eager'
    return c


I1=[10,20,30,40,11]
I2=[10,21,31,41,11]
I3=[10,22,32,42,11]


class StructureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def context(self, ids, samples=None, route=None):
        ids=torch.tensor([ids])
        return build_structure(ids, torch.zeros_like(ids) if samples is None else torch.tensor([samples]),
                               torch.ones_like(ids,dtype=torch.bool) if route is None else torch.tensor([route]),config())

    def test_text_does_not_break_history_and_targets_advance(self):
        c=self.context(I1+[7,8]+I2+[7]+I3+[10])
        self.assertEqual(c['histories'][0][0,-1].tolist(),I3[1:3])
        self.assertEqual(c['histories'][1][0,-1].tolist(),I2[1:3])
        self.assertTrue(c['history_valid'][1][0,-1])
        self.assertEqual(c['phase'][0,5].item(),-1)

    def test_partial_and_malformed_items_do_not_commit(self):
        c=self.context(I1+[10,21,31,7,11,10])
        self.assertEqual(c['histories'][0][0,-1].tolist(),I1[1:3])
        self.assertFalse(c['history_valid'][1][0,-1])
        self.assertEqual(c['phase'][0,7].item(),2)
        self.assertEqual(c['phase'][0,9].item(),-1)

    def test_packed_boundary_and_task_mask(self):
        ids=I1+I2+[10]
        c=self.context(ids,[0]*5+[1]*6)
        self.assertFalse(c['history_valid'][0][0,5])
        self.assertFalse(c['history_valid'][1][0,-1])
        c=self.context(ids,route=[False]*len(ids))
        self.assertFalse(c['itm_write'].any())
        self.assertEqual(c['phase'][0,4],4)

    def test_spm_does_not_read_other_items_or_text(self):
        c1=self.context(I1+[7]+I2)
        c2=self.context(I3+[8]+I2)
        a=build_memory_adapter(config(adapter_use_itm=False),16)
        h=torch.randn(1,11,16)
        torch.testing.assert_close(a(h,structure_context=c1)[:,-5:],a(h,structure_context=c2)[:,-5:])
        self.assertEqual(a(h,structure_context=c1)[0,5].abs().sum().item(),0)

    def test_switches_and_gradients(self):
        with self.assertRaises(ValueError):
            build_memory_adapter(config(adapter_use_spm=False,adapter_use_itm=False),16)
        for spm,itm in [(True,False),(False,True),(True,True)]:
            a=build_memory_adapter(config(adapter_use_spm=spm,adapter_use_itm=itm),16)
            c=self.context(I1+I2+[10])
            out=a(torch.randn(1,11,16),structure_context=c)
            out.square().sum().backward()
            self.assertEqual(a.spm is not None,spm)
            self.assertEqual(a.itm is not None,itm)
            if spm:self.assertGreater(a.spm.memories[0].weight.grad.abs().sum(),0)
            if itm:self.assertGreater(a.itm.memories[0].weight.grad.abs().sum(),0)

    def test_independent_memory_sizes_and_legacy_fallback(self):
        legacy = build_memory_adapter(config(), 16)
        self.assertEqual(legacy.spm.size, 31)
        self.assertEqual(legacy.itm.size, 31)

        split = build_memory_adapter(config(adapter_spm_memory_size=47,
                                            adapter_itm_memory_size=17), 16)
        self.assertEqual(split.spm.size, 47)
        self.assertEqual(split.itm.size, 17)
        self.assertEqual(split.spm_memory_size, 47)
        self.assertEqual(split.itm_memory_size, 17)

    def test_independent_structure_hidden_sizes_and_common_fallback(self):
        legacy = build_memory_adapter(config(adapter_hidden_size=4), 16)
        self.assertEqual(legacy.spm.memories[0].embedding_dim, 4)
        self.assertEqual(legacy.itm.memories[0].embedding_dim, 4)

        split_config = config(adapter_hidden_size=4, adapter_spm_hidden_size=3,
                              adapter_itm_hidden_size=7)
        split = build_memory_adapter(split_config, 16)
        self.assertEqual(split.spm.memories[0].embedding_dim, 3)
        self.assertEqual(split.spm.key_proj.in_features, 2 * 2 * 3)
        self.assertEqual(split.itm.memories[0].embedding_dim, 7)
        self.assertEqual(split.itm.key_proj.in_features, 2 * 2 * 7)
        self.assertEqual(split_config.adapter_spm_hidden_size, 3)
        self.assertEqual(split_config.adapter_itm_hidden_size, 7)

        common_fallback = build_memory_adapter(config(adapter_hidden_size=5,
                                                       adapter_spm_hidden_size=None,
                                                       adapter_itm_hidden_size=None), 16)
        self.assertEqual(common_fallback.spm.memories[0].embedding_dim, 5)
        self.assertEqual(common_fallback.itm.memories[0].embedding_dim, 5)

    def test_exact_itm_uses_itm_hidden_size_for_all_projection_widths(self):
        c = config(adapter_itm_mode='exact', adapter_itm_mapping=[0] + [-1] * 99,
                   adapter_itm_order=1, adapter_use_spm=False,
                   adapter_hidden_size=4, adapter_itm_hidden_size=7,
                   adapter_topk=2, adapter_itm_aux_loss_weight=.1)
        adapter = build_memory_adapter(c, 16)
        self.assertEqual(adapter.itm.memories[0].embedding_dim, 7)
        self.assertEqual(adapter.itm.key_proj.in_features, 2 * 7)
        self.assertEqual(adapter.itm.next_a_head.in_features, 2 * 7)

    def test_exact_itm_uses_compact_directed_source_slots(self):
        c = config(adapter_itm_mode='exact', adapter_itm_mapping=[0] + [-1] * 99,
                   adapter_itm_order=1,
                   adapter_use_spm=False, adapter_hidden_size=4)
        adapter = build_memory_adapter(c, 16)
        self.assertEqual(len(adapter.itm.memories), c.adapter_topk)
        self.assertTrue(all(table.num_embeddings == 1 for table in adapter.itm.memories))
        self.assertEqual(adapter.itm.key_proj.in_features, c.adapter_topk * c.adapter_hidden_size)
        self.assertIn('itm.memories.0.weight', adapter.state_dict())
        context = build_structure(torch.tensor([I1 + [10]]), torch.zeros(1, 6),
                                  torch.ones(1, 6, dtype=torch.bool), c)
        delta = adapter(torch.randn(1, 6, 16), structure_context=context)
        self.assertGreater(delta[0, -1].abs().sum().item(), 0)
        self.assertEqual(delta[0, 0:5].abs().sum().item(), 0)
        delta.sum().backward()
        for table in adapter.itm.memories:
            self.assertGreater(table.weight.grad.abs().sum().item(), 0)

    def test_exact_itm_unmatched_source_is_zero_and_default_is_legacy(self):
        c = config(adapter_itm_mode='exact', adapter_itm_mapping=[0] + [-1] * 99,
                   adapter_itm_order=1, adapter_use_spm=False)
        adapter = build_memory_adapter(c, 16)
        context = build_structure(torch.tensor([I2 + [10]]), torch.zeros(1, 6),
                                  torch.ones(1, 6, dtype=torch.bool), c)
        delta = adapter(torch.randn(1, 6, 16), structure_context=context)
        self.assertEqual(delta.abs().sum().item(), 0)
        legacy = build_memory_adapter(config(adapter_itm_mode='default'), 16)
        self.assertTrue(hasattr(legacy.itm, 'memories'))

    def test_future_independence_and_cached_logits(self):
        model=Qwen3ForCausalLM(config()).eval()
        ids=torch.tensor([I1+[7]+I2+[8]+I3])
        route=torch.tensor([[True]])
        with torch.no_grad():
            whole=model(ids,use_cache=False,itm_sequence_mask=route).logits
            modified=ids.clone();modified[0,-2]=49
            other=model(modified,use_cache=False,itm_sequence_mask=route).logits
            torch.testing.assert_close(whole[:,:-2],other[:,:-2])
            cache=None;outputs=[]
            for t in range(ids.shape[1]):
                o=model(ids[:,t:t+1],past_key_values=cache,use_cache=True,itm_sequence_mask=route)
                cache=o.past_key_values;outputs.append(o.logits)
            torch.testing.assert_close(whole,torch.cat(outputs,1),atol=2e-6,rtol=2e-5)

    def test_left_padding_cache_and_beam_generation(self):
        model=Qwen3ForCausalLM(config()).eval()
        ids=torch.tensor([[0,0]+I1+[10],[7,8]+I2+[10]])
        mask=(ids!=0).long();route=torch.tensor([[True],[True]])
        with torch.no_grad():
            whole=model.generate(ids,attention_mask=mask,itm_sequence_mask=route,max_new_tokens=3,
                num_beams=3,num_return_sequences=3,do_sample=False,use_cache=False)
            cached=model.generate(ids,attention_mask=mask,itm_sequence_mask=route,max_new_tokens=3,
                num_beams=3,num_return_sequences=3,do_sample=False,use_cache=True)
            self.assertTrue(torch.equal(whole,cached))

    def test_text_loss_reaches_spm_and_general_has_zero_delta(self):
        model=Qwen3ForCausalLM(config(adapter_use_itm=False)).train()
        for name,p in model.named_parameters():p.requires_grad_('memory_adapter' in name)
        ids=torch.tensor([I1+[7,8]])
        out=model(ids,sample_idx=torch.zeros_like(ids),itm_sequence_mask=torch.tensor([[False]]),use_cache=False)
        loss=torch.nn.functional.cross_entropy(out.logits[:,5],torch.tensor([8]))
        loss.backward()
        grad=model.model.layers[0].memory_adapter.spm.memories[0].weight.grad
        self.assertGreater(grad.abs().sum(),0)
        ids=torch.tensor([[5,6,7]])
        c=prepare_structure(ids,None,torch.tensor([[False]]),None,model.config)
        self.assertEqual(model.model.layers[0].memory_adapter(torch.randn(1,3,16),structure_context=c).abs().sum(),0)

    def test_cache_reorder_repeat_and_crop(self):
        cache=StructuredDynamicCache()
        cache.structure_ids=torch.tensor([I1,I2]);cache.structure_samples=torch.zeros(2,5,dtype=torch.long)
        cache.structure_routes=torch.ones(2,5,dtype=torch.bool)
        cache.update(torch.zeros(2,2,5,8),torch.zeros(2,2,5,8),0)
        cache.reorder_cache(torch.tensor([1,0]))
        self.assertEqual(cache.structure_ids[0].tolist(),I2)
        cache.batch_repeat_interleave(2)
        self.assertEqual(cache.structure_ids.shape[0],4)
        cache.batch_select_indices(torch.tensor([3,0]))
        self.assertEqual(cache.structure_ids[0].tolist(),I1)
        cache.crop(3)
        self.assertEqual(cache.structure_ids.shape[1],3)

    def test_gradient_checkpointing_matches_plain_backward(self):
        a=Qwen3ForCausalLM(config()).train()
        b=Qwen3ForCausalLM(config()).train();b.load_state_dict(a.state_dict())
        b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ids=torch.tensor([I1+[7]+I2+[10]])
        kw=dict(sample_idx=torch.zeros_like(ids),itm_sequence_mask=torch.tensor([[True]]),use_cache=False)
        for m in (a,b):
            m(ids,**kw).logits.square().mean().backward()
        for (name,p1),(_,p2) in zip(a.named_parameters(),b.named_parameters()):
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad,p2.grad,atol=1e-6,rtol=1e-5,msg=name)

    def test_task_mask_packing_and_truncation(self):
        from types import SimpleNamespace
        from onerec_llm.data.qwen3_dataset import Qwen3ChatCompletionDataset
        ds=Qwen3ChatCompletionDataset.__new__(Qwen3ChatCompletionDataset)
        ds.max_length=8;ds.max_sample_length=8;ds.cut_to_pad=True;ds.kwargs={}
        ds.tokenizer=SimpleNamespace(pad_token_id=0)
        def process(sample):
            ids=torch.tensor([I1])
            return dict(input_ids=ids,attention_mask=torch.ones_like(ids),loss_mask=torch.ones_like(ids),
                        position_ids=torch.arange(5)[None],itemic_id_mask=torch.ones_like(ids))
        ds._process_completion=process
        sample={'json':{'segments':[]},'epoch_idx':0}
        rec=ds._process(sample,'RecIF_VideoRec_Pretrain')
        profile=ds._process(sample,'RecIF_UserProfile_Pretrain')
        packed=ds._packing([rec,profile])
        self.assertTrue(packed['itm_sequence_mask'][packed['sample_idx']==0].all())
        self.assertFalse(packed['itm_sequence_mask'][packed['sample_idx']==1].any())
        self.assertFalse(packed['itm_sequence_mask'][packed['sample_idx']==-1].any())
        self.assertEqual(int((packed['sample_idx']==1).sum()),3)
        c=build_structure(packed['input_ids'],packed['sample_idx'],packed['itm_sequence_mask'],config())
        self.assertFalse(c['history_valid'][0][0,5])

    def test_training_requires_explicit_structure_inputs(self):
        m=Qwen3ForCausalLM(config()).train();ids=torch.tensor([I1])
        with self.assertRaises(ValueError):m(ids,use_cache=False)
        with self.assertRaises(ValueError):m(ids,itm_sequence_mask=torch.tensor([[True]]),use_cache=False)

    def test_stage1_plain_config_defaults(self):
        from transformers import Qwen3Config as PlainConfig
        c=PlainConfig(hidden_size=16)
        c.adapter_sid_begin_id=10;c.adapter_sid_end_id=11
        c.adapter_sid_code_ranges=[[20,29],[30,39],[40,49]]
        c.adapter_type='engram_structure'
        self.assertIsNotNone(build_memory_adapter(c,16).spm)

    def test_converter_restores_exact_run_switches(self):
        import json
        source=Path(__file__).resolve().parents[1]/'tools/model_converter/convert_checkpoint_to_hf.py'
        spec=importlib.util.spec_from_file_location('stage5_cfg_converter',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);ckpt=run/'step1'/'global_step1';ckpt.mkdir(parents=True)
            out=ckpt/'converted';out.mkdir()
            (out/'config.json').write_text(json.dumps({'adapter_type':None}))
            actual={'adapter_type':'engram_structure','adapter_use_spm':False,'adapter_use_itm':True,
                    'adapter_itm_order':2,'adapter_spm_hidden_size':3,'adapter_itm_hidden_size':7}
            (run/'structure_adapter_config.json').write_text(json.dumps(actual))
            self.assertTrue(module.restore_structure_config(ckpt,out))
            self.assertEqual(json.loads((out/'config.json').read_text()),actual)

    @unittest.skipUnless(os.environ.get('STAGE5_GPU_TEST')=='1', 'opt-in small CUDA/FlashAttention test')
    def test_flash_packed_training_matches_separate_samples(self):
        c=config();c._attn_implementation='flash_attention_2'
        m=Qwen3ForCausalLM(c).to(device='cuda',dtype=torch.bfloat16).train()
        first=I1+[10];second=I2+[10]
        ids=torch.tensor([first+second],device='cuda')
        samples=torch.tensor([[0]*6+[1]*6],device='cuda')
        route=torch.ones_like(ids,dtype=torch.bool)
        positions=torch.tensor([list(range(6))*2],device='cuda')
        cu=torch.tensor([0,6,12],device='cuda',dtype=torch.int32)
        output=m(ids,sample_idx=samples,itm_sequence_mask=route,position_ids=positions,cu_seqlens=cu,use_cache=False).logits
        separate=[]
        for seq in (first,second):
            x=torch.tensor([seq],device='cuda')
            separate.append(m(x,sample_idx=torch.zeros_like(x),itm_sequence_mask=torch.ones_like(x,dtype=torch.bool),use_cache=False).logits)
        torch.testing.assert_close(output,torch.cat(separate,1),atol=.003,rtol=.03)
        loss=output.float().square().mean();loss.backward()
        self.assertGreater(m.model.layers[0].memory_adapter.itm.memories[0].weight.grad.abs().sum(),0)
        opt=torch.optim.AdamW([p for n,p in m.named_parameters() if 'memory_adapter' in n],lr=.001)
        before=m.model.layers[0].memory_adapter.itm.memories[0].weight.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before,m.model.layers[0].memory_adapter.itm.memories[0].weight))

    def test_remote_bundle_roundtrip(self):
        converter_path=Path(__file__).resolve().parents[1]/'tools/model_converter/convert_checkpoint_to_hf.py'
        spec=importlib.util.spec_from_file_location('stage5_converter',converter_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=Qwen3ForCausalLM(config()).eval()
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            module.copy_remote_code_bundle(path);module.update_remote_code_config(path)
            from transformers import AutoModelForCausalLM
            loaded=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True).eval()
            ids=torch.tensor([I1+[10]])
            with torch.no_grad():
                a=model(ids,itm_sequence_mask=torch.tensor([[True]]),use_cache=False).logits
                b=loaded(ids,itm_sequence_mask=torch.tensor([[True]]),use_cache=False).logits
            torch.testing.assert_close(a,b)

    def test_remote_bundle_contains_lngram_dependencies(self):
        converter_path=Path(__file__).resolve().parents[1]/'tools/model_converter/convert_checkpoint_to_hf.py'
        spec=importlib.util.spec_from_file_location('stage5_lngram_converter',converter_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        bundled=module._build_remote_memory_adapter_module()
        self.assertIn('class LatentSymbolMemory',bundled)
        self.assertIn('class StructuredLngramMemoryAdapter',bundled)
        self.assertNotIn('from .lngram import',bundled)
        self.assertNotIn('from .lngram_structure import',bundled)

    def test_remote_bundle_has_no_initialization_dependency(self):
        converter_path=Path(__file__).resolve().parents[1]/'tools/model_converter/convert_checkpoint_to_hf.py'
        spec=importlib.util.spec_from_file_location('stage5_init_converter',converter_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as path:
            module.copy_remote_code_bundle(path)
            self.assertFalse((Path(path)/'initialization.py').exists())
            bundled=(Path(path)/'memory_adapter.py').read_text()
            self.assertNotIn('def initialize_embedding', bundled)
            self.assertNotIn('from .initialization import', bundled)



if __name__=='__main__':unittest.main()
