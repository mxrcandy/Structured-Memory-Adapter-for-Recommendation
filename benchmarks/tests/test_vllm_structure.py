"""Tiny opt-in real vLLM engine test; no trained checkpoint is required."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'benchmarks'), str(ROOT/'pretrain'), str(ROOT/'benchmarks/scripts/ray-vllm')]
from onerec_llm.models.qwen3.configuration_qwen3 import Qwen3Config
from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from onerec_llm.models.memory_adapter.structure import build_structure, configure_structure_tokens
from benchmark.vllm_structure import scheduler_context, context_to_kwargs, kwargs_to_context, set_itm_route, engram_lookup_ids


class ContextTests(unittest.TestCase):
    def test_exact_itm_scheduler_matches_hf_chunked_and_packed(self):
        from onerec_llm.models.memory_adapter import build_memory_adapter
        c = Qwen3Config(hidden_size=16, adapter_type='engram_structure', adapter_use_spm=False,
                       adapter_itm_mode='exact', adapter_itm_mapping=[0]+[-1]*99,
                       adapter_hidden_size=4, adapter_itm_order=1, adapter_fixed_context_gate=.5,
                       adapter_sid_begin_id=10, adapter_sid_end_id=11,
                       adapter_sid_code_ranges=[[20,29],[30,39],[40,49],[50,59]])
        adapter = build_memory_adapter(c, 16)
        seqs = [[10,20,30,40,50,11,10,21,31,41,51,11,10],
                [10,21,31,41,51,11,10]]
        for lengths, queries in (([13,7], [13,7]), ([13,7], [1,1]), ([7,5], [2,2])):
            for route in (True, False):
                result = scheduler_context(seqs, lengths, queries, c, route, 'cpu')
                context = kwargs_to_context({k:v[None] for k,v in result.items()}, c)
                hidden = torch.randn(1, sum(queries), 16)
                got = adapter(hidden, structure_context=context)
                expected, offset = [], 0
                for seq, length, query in zip(seqs, lengths, queries):
                    ids = torch.tensor([seq[:length]])
                    full = build_structure(ids, torch.zeros_like(ids), torch.full_like(ids,route,dtype=torch.bool), c)
                    h = torch.zeros(1, length, 16)
                    h[:, -query:] = hidden[:, offset:offset+query]
                    expected.append(adapter(h, structure_context=full)[:, -query:])
                    offset += query
                torch.testing.assert_close(got, torch.cat(expected, 1))
        self.assertGreater(adapter(torch.randn(1,20,16), structure_context=kwargs_to_context(
            {k:v[None] for k,v in scheduler_context(seqs,[13,7],[13,7],c,True,'cpu').items()},c)
        ).abs().sum().item(), 0)

    def test_engram_lookup_uses_complete_history(self):
        c = Qwen3Config(adapter_type='engram', adapter_memory_size=31,
                        adapter_ngram_order=3, adapter_topk=2)
        history = [[10, 20, 30, 40]]
        result = scheduler_context(history, [4], [1], c, False, 'cpu')
        expected = engram_lookup_ids(torch.tensor(history), c)[0, -1]
        torch.testing.assert_close(result['vllm_memory_lookup_ids'][0], expected)
        self.assertFalse(torch.equal(expected, engram_lookup_ids(torch.tensor([[40]]), c)[0, 0]))

    def test_all_memory_types_use_scheduler_history_transport(self):
        base = dict(adapter_ngram_order=3, adapter_memory_size=31, adapter_topk=2,
                    adapter_sid_begin_id=10, adapter_sid_end_id=11,
                    adapter_sid_code_ranges=[[20, 29], [30, 39], [40, 49]],
                    adapter_use_spm=True, adapter_use_itm=True, adapter_itm_order=1)
        for adapter_type in ('engram', 'lngram', 'engram_structure', 'lngram_structure'):
            c = Qwen3Config(adapter_type=adapter_type, **base)
            kwargs = dict(histories=[[10, 20, 30, 40, 11]], seq_lens=[5], query_lens=[1],
                          config=c, route=True, device='cpu')
            if adapter_type in ('lngram', 'lngram_structure'):
                kwargs.update(block_tables=[[7]], block_size=16)
            result = scheduler_context(**kwargs)
            self.assertEqual(result['vllm_memory_lookup_ids'].shape[0] if adapter_type.startswith('engram') else 1, 1)
            if adapter_type.startswith('lngram'):
                self.assertIn('structure_symbol_slots', result)
            if adapter_type.endswith('structure'):
                self.assertIn('structure_phase', result)

    def test_scheduler_queries_match_independent_sequences(self):
        c = Qwen3Config(adapter_ngram_order=5, adapter_sid_begin_id=10, adapter_sid_end_id=11,
                       adapter_sid_code_ranges=[[20,29],[30,39],[40,49]])
        sequences = [[10,21,32,43,11,7,10,22,33,44,11,8,10],
                     [7,10,24,35,46,11,9,10,25], [10,21,32,43,11]]
        for route in (False, True):
            for queries in ([13,9,5], [1,1,1], [3,4,2]):
                result = scheduler_context(sequences, list(map(len,sequences)), queries, c, route, 'cpu')
                expected = []
                offset = 0
                for history, query in zip(sequences,queries):
                    ids = torch.tensor([history])
                    context = build_structure(ids, torch.zeros_like(ids), torch.full_like(ids,route,dtype=torch.bool), c)
                    # Scheduler positions index the concatenated full histories.
                    context['history_positions'] = [p + offset for p in context['history_positions']]
                    expected.append({k:v[0,-query:] for k,v in context_to_kwargs(context).items()})
                    offset += len(history)
                for key in result:
                    want = torch.cat([x[key] for x in expected])
                    # Invalid lookup keys are arbitrary; their masks must agree.
                    mask = torch.ones(result[key].shape[0], dtype=torch.bool)
                    if key.startswith('structure_window_'):
                        mask = result['structure_phase'] >= int(key.rsplit('_', 1)[1]) - 1
                    elif key.startswith('structure_history_'):
                        mask = result['structure_valid_' + key.rsplit('_', 1)[1]]
                    elif key.startswith('structure_position_'):
                        mask = result['structure_valid_' + key.rsplit('_', 1)[1]]
                    torch.testing.assert_close(result[key][mask], want[mask])
        # Chunked prefill must not see the remainder of a prompt.
        truncated = scheduler_context(sequences,[8,7,4],[3,2,4],c,True,'cpu')
        changed = [s[:n]+[99]*4 for s,n in zip(sequences,[8,7,4])]
        other = scheduler_context(changed,[8,7,4],[3,2,4],c,True,'cpu')
        for key in truncated:
            torch.testing.assert_close(truncated[key],other[key])


@unittest.skipUnless(os.environ.get('STAGE5_GPU_TEST') == '1','opt-in real CUDA/vLLM engine')
class NativeEngineTests(unittest.TestCase):
    adapter_type = 'engram_structure'
    itm_mode = 'default'
    aux_weight = 0.
    memory_heads = 1

    def test_export_batch_decode_beam_and_routing(self):
        from utils.generator import VllmWorker
        from vllm import SamplingParams
        torch.manual_seed(17)
        tok = AutoTokenizer.from_pretrained(ROOT/'hf_models/Qwen3-1.7B_itemic',trust_remote_code=True)
        c = Qwen3Config(vocab_size=len(tok),hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                        num_attention_heads=2,num_key_value_heads=2,head_dim=32,
                        adapter_type=self.adapter_type,adapter_lngram_bits_per_route=2,
                        adapter_hidden_size=8,adapter_memory_size=31,
                        adapter_topk=self.memory_heads,adapter_ngram_order=5,adapter_itm_order=2,
                        adapter_fixed_context_gate=.5,adapter_use_conv=False,
                        pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        configure_structure_tokens(c,tok)
        if self.itm_mode in ('exact', 'successor_prior', 'successor_prior_learned'):
            c.adapter_itm_mode = self.itm_mode
            c.adapter_itm_order = 1
            base_a, base_b = [hi-lo+1 for lo,hi in c.adapter_sid_code_ranges[:2]]
            c.adapter_itm_mapping = [-1] * (base_a * base_b)
            c.adapter_itm_mapping[base_b + 2] = 0
            c.adapter_itm_aux_loss_weight = self.aux_weight
            c.adapter_itm_aux_head = self.aux_weight > 0
            if self.itm_mode in ('successor_prior', 'successor_prior_learned'):
                c.adapter_itm_prior_counts = [[[4,3],[7,1]]]
        c._attn_implementation='sdpa'
        hf = Qwen3ForCausalLM(c).eval().to(dtype=torch.bfloat16,device='cuda')
        with torch.no_grad():
            for name,p in hf.named_parameters():
                if 'memory_adapter' in name and ('memories' in name or '.tables.' in name):
                    p.normal_(0,.4)
                if self.itm_mode == 'successor_prior' and '.itm.prior_proj.weight' in name:
                    p.normal_(0,.4)
        spec=importlib.util.spec_from_file_location('converter', ROOT/'pretrain/tools/model_converter/convert_checkpoint_to_hf.py')
        converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
        item='<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>'
        prompts=[(item+'文字间隔')*4+'推荐：<|sid_begin|>',
                 item+'下一个：<|sid_begin|><s_a_4><s_b_5>', '2+2=']
        with tempfile.TemporaryDirectory(prefix='stage5-vllm-') as directory:
            hf.save_pretrained(directory);tok.save_pretrained(directory)
            converter.copy_remote_code_bundle(directory);converter.update_remote_code_config(directory)
            gpu_ids = [int(x) for x in os.environ.get('STAGE5_PHYSICAL_GPUS','0').split(',')]
            worker=VllmWorker(0,directory,gpu_ids,tensor_parallel_size=len(gpu_ids),
                              dtype='bfloat16',max_model_len=128,gpu_memory_utilization=.08,
                              enable_optimizations=True,num_gpu_blocks_override=64,
                              max_num_seqs=8,max_num_batched_tokens=32,max_logprobs=8)
            self.assertTrue(worker.native_structure)
            native=worker.llm.llm_engine.model_executor.driver_worker.model_runner.model
            native_weights=dict(native.named_parameters())
            for name,p in hf.named_parameters():
                if 'memory_adapter' in name:
                    self.assertIn(name,native_weights)
                    torch.testing.assert_close(native_weights[name],p,rtol=0,atol=0)
            print('ADAPTER_WEIGHTS_EXACT',flush=True)

            def reference(ids,route):
                with torch.inference_mode():
                    result=hf(torch.tensor([ids],device='cuda'),itm_sequence_mask=torch.tensor([route],device='cuda'),
                              use_cache=False,chunked_loss_computer=False)
                    return result.logits[0].float().log_softmax(-1)

            for route in (True,False,True):
                set_itm_route(worker.llm,route)
                outputs=worker.llm.generate(prompts,SamplingParams(temperature=0,max_tokens=3,
                                                                   logprobs=1,ignore_eos=True),use_tqdm=False)
                for output in outputs:
                    history=list(output.prompt_token_ids)
                    for token,logprobs in zip(output.outputs[0].token_ids,output.outputs[0].logprobs):
                        expected=reference(history,route)[-1]
                        self.assertAlmostEqual(float(expected[token]),logprobs[token].logprob,delta=.04)
                        history.append(token)
                repeat=worker.llm.generate(prompts,SamplingParams(temperature=0,max_tokens=3,ignore_eos=True),use_tqdm=False)
                self.assertEqual([list(x.outputs[0].token_ids) for x in outputs],
                                 [list(x.outputs[0].token_ids) for x in repeat])
            # Nonzero route effect: ITM must participate, not just load its weights.
            ids=tok.encode(prompts[0])
            self.assertGreater(float((reference(ids,True)[-1]-reference(ids,False)[-1]).abs().max()),.01)
            print('BATCH_DECODE_LOGPROBS_PREFIX_CACHE_AND_ROUTE_OK',flush=True)
            texts,scores,stats=worker.generate_batch({'a':prompts[0],'b':prompts[1]},
                dict(use_beam_search=True,beam_width=2,max_tokens=3,itm_sequence=True),worker_batch_size=2)
            for key in ('a','b'):
                self.assertEqual(len(texts[key]),2)
                self.assertEqual(len(scores[key]),2)
                self.assertEqual(stats[key]['backend'],'vllm_transformers_memory')
            print('BATCH_BEAM_OK',flush=True)
            # Candidate expansion n>1 uses native request/KV branching.
            set_itm_route(worker.llm,True)
            forked=worker.llm.generate(prompts[:2],SamplingParams(n=2,temperature=1.0,max_tokens=3,
                                                               logprobs=1,ignore_eos=True),use_tqdm=False)
            for output in forked:
                for candidate in output.outputs:
                    history=list(output.prompt_token_ids)
                    for token,logprobs in zip(candidate.token_ids,candidate.logprobs):
                        # vLLM 0.7.3 reports post-temperature logprobs; use T=1 for this comparison.
                        self.assertAlmostEqual(float(reference(history,True)[-1,token]),
                                               logprobs[token].logprob,delta=.04)
                        history.append(token)
            print('FORKED_SAMPLING_OK',flush=True)


class LngramNativeEngineTests(NativeEngineTests):
    adapter_type = 'lngram_structure'


class ExactNativeEngineTests(NativeEngineTests):
    itm_mode = 'exact'


class ExactAuxNativeEngineTests(ExactNativeEngineTests):
    aux_weight = .1
    memory_heads = 4


class SuccessorPriorNativeEngineTests(NativeEngineTests):
    itm_mode = 'successor_prior'
    memory_heads = 4


class SuccessorPriorLearnedNativeEngineTests(NativeEngineTests):
    itm_mode = 'successor_prior_learned'
    memory_heads = 4


if __name__=='__main__':
    unittest.main()
