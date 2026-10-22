"""Successor prior contracts: shared embeddings, causal masks and persistence."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from test_engram_structure import config, I1, I2
from onerec_llm.models.memory_adapter.structure import configure_successor_prior, SuccessorPriorBranch, SuccessorPriorLearnedBranch, restore_exact_itm_for_resume
from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def prior_config(**kwargs):
    values=dict(adapter_itm_mode='successor_prior',adapter_itm_order=1,
                adapter_use_spm=False,adapter_itm_mapping=[0]+[-1]*99,
                adapter_itm_prior_counts=[[[1,3],[2,1]]])
    values.update(kwargs)
    return config(**values)


class PriorTests(unittest.TestCase):
    def test_miss_route_boundary_and_resume_counts(self):
        from onerec_llm.models.memory_adapter import build_memory_adapter
        from onerec_llm.models.memory_adapter.structure import build_structure
        c=prior_config();adapter=build_memory_adapter(c,16)
        ids=torch.tensor([I1+I2+I2])
        samples=torch.tensor([[0]*10+[1]*5])
        ctx=build_structure(ids,samples,torch.ones_like(ids,dtype=torch.bool),c)
        delta=adapter(torch.randn(1,15,16),structure_context=ctx,itm_token_embeddings=torch.randn(10,16))
        self.assertGreater(delta[:,5].abs().sum(),0)
        self.assertEqual(delta[:,:5].abs().sum(),0)
        self.assertEqual(delta[:,6:].abs().sum(),0)
        ctx['itm_write'][:]=False
        self.assertEqual(adapter(torch.randn(1,15,16),structure_context=ctx,itm_token_embeddings=torch.randn(10,16)).abs().sum(),0)
        with tempfile.TemporaryDirectory() as d:
            saved={k:v for k,v in c.to_dict().items() if k.startswith('adapter_')}
            (Path(d)/'structure_adapter_config.json').write_text(json.dumps(saved))
            restore_exact_itm_for_resume(prior_config(),d)
            with self.assertRaises(ValueError):
                restore_exact_itm_for_resume(prior_config(adapter_itm_prior_counts=[[[1,4],[2,1]]]),d)
            for key,value in [('adapter_use_alpha_layer',True),('adapter_fixed_context_gate',.7)]:
                changed=dict(saved);changed[key]=value
                (Path(d)/'structure_adapter_config.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,key):
                    restore_exact_itm_for_resume(prior_config(),d)
            changed=dict(saved);changed['adapter_itm_hidden_size']=8
            (Path(d)/'structure_adapter_config.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'adapter_itm_hidden_size'):
                restore_exact_itm_for_resume(prior_config(adapter_itm_hidden_size=7),d)

    def test_weighted_embedding_mask_and_detached_gradient(self):
        c=prior_config();branch=SuccessorPriorBranch(c,16)
        emb=torch.randn(10,16,requires_grad=True)
        hidden=torch.randn(1,3,16)
        keys=torch.tensor([[[c.adapter_sid_code_ranges[0][0],c.adapter_sid_code_ranges[1][0]]]*3])
        valid=torch.tensor([[True,False,True]])
        seen=[]
        handle=branch.prior_proj.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach()))
        out=branch(hidden,keys,valid,emb);handle.remove()
        torch.testing.assert_close(seen[0][0,0],.75*emb[1]+.25*emb[2])
        self.assertEqual(out[0,1].abs().sum(),0)
        out.sum().backward()
        self.assertIsNone(emb.grad)
        self.assertIsNotNone(branch.prior_proj.weight.grad)
        self.assertFalse(any('memor' in k for k in branch.state_dict()))

    def test_learned_prior_has_topk_a_tables_and_gradient(self):
        c=prior_config(adapter_itm_mode='successor_prior_learned', adapter_topk=4)
        branch=SuccessorPriorLearnedBranch(c,16)
        self.assertEqual(len(branch.memories),4)
        self.assertEqual(tuple(branch.memories[0].weight.shape),(10,4))
        with torch.no_grad():
            for head,table in enumerate(branch.memories):
                table.weight.zero_();table.weight[1].fill_(head+1);table.weight[2].fill_(head+3)
        hidden=torch.randn(1,3,16,requires_grad=True)
        keys=torch.tensor([[[20,30]]*3]); valid=torch.tensor([[True,False,True]])
        captured=[];hook=branch.value_proj.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
        out=branch(hidden,keys,valid);hook.remove()
        torch.testing.assert_close(captured[0][0,0],torch.tensor([1.5]*4+[2.5]*4+[3.5]*4+[4.5]*4))
        out.sum().backward()
        self.assertTrue(all(t.weight.grad is not None for t in branch.memories))
        self.assertEqual(tuple(branch.key_proj.weight.shape),(16,16))

    def test_prior_branches_use_itm_specific_hidden_size(self):
        fixed = SuccessorPriorBranch(prior_config(adapter_hidden_size=4,
                                                    adapter_itm_hidden_size=7), 16)
        self.assertEqual(tuple(fixed.prior_proj.weight.shape), (7, 16))
        self.assertEqual(tuple(fixed.key_proj.weight.shape), (16, 7))

        learned = SuccessorPriorLearnedBranch(prior_config(
            adapter_itm_mode='successor_prior_learned', adapter_topk=2,
            adapter_hidden_size=4, adapter_itm_hidden_size=7), 16)
        self.assertEqual(tuple(learned.memories[0].weight.shape), (10, 7))
        self.assertEqual(learned.key_proj.in_features, 14)

    def test_artifact_validation_and_saved_config(self):
        c=prior_config(adapter_itm_mapping=None,adapter_itm_prior_counts=None)
        artifact=dict(format='itm_successor_prior_v1',source_levels=['a','b'],target_level='a',
            sid_base=10,minimum_edge_frequency=3,self_transition_policy='include',
            distribution_policy='all_successors',source_count=1,exclude_last_item=True,
            sources=[dict(source=[0,0],total_count=4,successors=[dict(a=1,count=3),dict(a=2,count=1)])])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'prior.json';p.write_text(json.dumps(artifact))
            configure_successor_prior(c,p,pattern_min_frequency=3)
            self.assertEqual(c.adapter_itm_prior_counts,[[[1,3],[2,1]]])
            configure_successor_prior(c,None,pattern_min_frequency=3)
            artifact['sources'][0]['successors'][0]['count']=2
            p.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):configure_successor_prior(c,p)

    def test_hf_cache_and_save_reload(self):
        torch.manual_seed(18)
        c=prior_config();model=Qwen3ForCausalLM(c).eval()
        ids=torch.tensor([I1+I2])
        with torch.no_grad():
            full=model(ids,use_cache=False,itm_sequence_mask=torch.tensor([True])).logits
            first=model(ids[:,:5],use_cache=True,itm_sequence_mask=torch.tensor([True]))
            rest=model(ids[:,5:],past_key_values=first.past_key_values,use_cache=True).logits
        torch.testing.assert_close(full[:,5:],rest,atol=1e-5,rtol=1e-5)
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            loaded=Qwen3ForCausalLM.from_pretrained(d).eval()
            with torch.no_grad():out=loaded(ids,use_cache=False,itm_sequence_mask=torch.tensor([True])).logits
            torch.testing.assert_close(full,out)

    def test_stage2_dcp_conversion_sft_and_strict_export(self):
        from tools.model_converter import convert_checkpoint_to_hf as converter
        from transformers import AutoModelForCausalLM, AutoModel
        from safetensors.torch import load_file, save_file
        import torch.distributed.checkpoint as dcp
        model=Qwen3ForCausalLM(prior_config())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'base';source.mkdir()
            cfg=model.config.to_dict()
            cfg.update(adapter_itm_mode='default',adapter_itm_mapping=None,adapter_itm_prior_counts=None)
            (source/'config.json').write_text(json.dumps(cfg))
            for stage in ('stage2','sft'):
                run=root/stage;run.mkdir()
                checkpoint=run/('dcp' if stage=='stage2' else 'model.pth')
                if stage=='stage2':dcp.save({'app':{'model':model.state_dict()}},checkpoint_id=checkpoint)
                else:torch.save({'model':model.state_dict()},checkpoint)
                (run/'structure_adapter_config.json').write_text(json.dumps({k:v for k,v in model.config.to_dict().items() if k.startswith('adapter_')}))
                output=run/'export'
                result=subprocess.run([sys.executable,converter.__file__,'--checkpoint_dir',str(checkpoint),
                    '--output_dir',str(output),'--source_hf_model_path',str(source),'--bundle_remote_code','--dtype','bf16'],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                restored,info=AutoModelForCausalLM.from_pretrained(output,trust_remote_code=True,local_files_only=True,
                    output_loading_info=True,attn_implementation='eager')
                self.assertEqual(info['missing_keys'],[]);self.assertEqual(info['unexpected_keys'],[])
                self.assertEqual(restored.config.adapter_itm_prior_counts,model.config.adapter_itm_prior_counts)
                for k,v in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[k],v.bfloat16().to(restored.state_dict()[k].dtype))
                decoder=AutoModel.from_config(restored.config,trust_remote_code=True).eval()
                decoder.load_state_dict(restored.model.state_dict(),strict=True)
                ids=torch.tensor([I1+I2]);restored.train()
                out=restored(ids,sample_idx=torch.zeros_like(ids),itm_sequence_mask=torch.tensor([True]),use_cache=False)
                self.assertIsNone(out.itm_aux_stats)
                torch.nn.functional.cross_entropy(out.logits[:,5].float(),ids[:,6]).backward()
                torch.optim.AdamW(restored.parameters(),lr=.001).step()
                # Missing prior weights must fail even though generic adapters permit new params.
                index=json.loads((output/'model.safetensors.index.json').read_text())
                key=next(k for k in index['weight_map'] if '.itm.prior_proj.' in k)
                file=output/index['weight_map'][key];state=load_file(file)
                state.pop(key)
                save_file(state,file,metadata={'format':'pt'})
                with self.assertRaisesRegex(ValueError,'successor prior weights'):
                    converter.validate_itm_aux_export(output)
                model,source=restored,output

    def test_bin_export_validation(self):
        from tools.model_converter import convert_checkpoint_to_hf as converter
        model=Qwen3ForCausalLM(prior_config())
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            (output/'config.json').write_text(model.config.to_json_string())
            converter._convert_state_dict_to_shards(model.state_dict(),d,use_safetensor=False)
            converter.validate_itm_aux_export(d)
            index=json.loads((output/'model.bin.index.json').read_text())
            key=next(k for k in index['weight_map'] if '.itm.prior_proj.' in k)
            shard=output/index['weight_map'][key]
            state=torch.load(shard,weights_only=True);state.pop(key);torch.save(state,shard)
            with self.assertRaisesRegex(ValueError,'successor prior weights'):
                converter.validate_itm_aux_export(d)

    def test_learned_prior_export_shapes(self):
        from tools.model_converter import convert_checkpoint_to_hf as converter
        model=Qwen3ForCausalLM(prior_config(adapter_itm_mode='successor_prior_learned', adapter_topk=4,
                                             adapter_hidden_size=4, adapter_itm_hidden_size=7))
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            converter.validate_itm_aux_export(d)
            restored=Qwen3ForCausalLM.from_pretrained(d)
            self.assertEqual(restored.model.layers[0].memory_adapter.itm.memories[0].embedding_dim,7)

    @unittest.skipUnless(os.environ.get('PRIOR_GPU_TEST')=='1','opt-in CUDA FSDP')
    def test_fsdp_checkpointing_real_training_step(self):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from onerec_llm.training.distributed import shard_model,load_from_full_model_state_dict
        from onerec_llm.training.activations import set_activation_checkpointing
        from recipes.train_qwen3 import compute_forward_backward
        from onerec_llm.losses import CrossEntropyLoss,ChunkedLossComputer
        with tempfile.TemporaryDirectory() as d:
            torch.cuda.set_device(0)
            dist.init_process_group('nccl',init_method='file://'+str(Path(d)/'rdzv'),rank=0,world_size=1)
            try:
                cfg=prior_config();cfg.use_cache=False
                reference=Qwen3ForCausalLM(cfg);full=reference.state_dict()
                with torch.device('meta'):model=Qwen3ForCausalLM(cfg)
                set_activation_checkpointing(model,auto_wrap_policy=Qwen3ForCausalLM.wrap_modules)
                shard_model(model,cpu_offload=False,dp_mesh=init_device_mesh('cuda',(1,)),fp32_weight=True)
                with self.assertRaisesRegex(ValueError,'prior weights mismatch'):
                    load_from_full_model_state_dict(model,{k:v for k,v in full.items() if '.itm.prior_proj.' not in k},require_itm_prior=True)
                load_from_full_model_state_dict(model,full,require_itm_prior=True)
                model.model.rotary_emb.inv_freq=reference.model.rotary_emb.inv_freq.cuda()
                for name,p in model.named_parameters():p.requires_grad_('memory_adapter' in name)
                ids=torch.tensor([I1+I2],device='cuda')
                batch=dict(input_ids=ids,sample_idx=torch.zeros_like(ids),itm_sequence_mask=torch.tensor([True],device='cuda'),loss_mask=torch.ones_like(ids))
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
                ce=CrossEntropyLoss(return_token_loss=True,shift_labels=False)
                proj=model.model.layers[0].memory_adapter.itm.prior_proj.weight
                for chunked in (False,True):
                    optimizer.zero_grad(set_to_none=True);model.config.chunked_loss_computer=chunked;model.chunked_loss_computer=chunked
                    compute=ChunkedLossComputer(model.lm_head,ce,minibatch_size=3,shift_labels=False).forward_and_backward if chunked else ce
                    loss,_,metrics=compute_forward_backward(model,batch,compute,ce,SimpleNamespace(
                        start_optimize_embedding_index=0,max_grad_norm=0,use_chunked_loss_computer=chunked),None,optimizer)
                    self.assertTrue(torch.isfinite(loss));self.assertEqual(metrics,{})
                    self.assertGreater(proj.grad.to_local().abs().sum(),0)
                    before=proj.to_local().detach().clone();optimizer.step()
                    self.assertFalse(torch.equal(before,proj.to_local()))
            finally:dist.destroy_process_group()


if __name__=='__main__':unittest.main()
