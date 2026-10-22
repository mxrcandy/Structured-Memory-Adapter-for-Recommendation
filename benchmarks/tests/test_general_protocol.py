from types import SimpleNamespace

from benchmarks.lm_eval import protocol


def test_template_passes_explicit_nonthinking():
    class Tokenizer:
        def apply_chat_template(self, history, **kwargs):
            assert kwargs['enable_thinking'] is False
            assert kwargs['tokenize'] is False
            return 'assistant\n<think>\n\n</think>\n\n'
    assert '</think>' in protocol.render_chat(SimpleNamespace(tokenizer=Tokenizer()), [])


def test_task_overrides_preserve_stops_and_ifeval_prompt():
    class Task:
        def __init__(self, name):
            self.config = {'task': name, 'output_type': 'generate_until',
                           'doc_to_text': 'original', 'generation_kwargs': {'until': ['STOP'], 'max_gen_toks': 256}}
        def get_config(self, key):
            return self.config.get(key)
        def set_config(self, key, value, update=False):
            if update:
                self.config[key].update(value)
            else:
                self.config[key] = value
    gsm, ifeval = Task('gsm8k'), Task('ifeval')
    protocol.configure_tasks({'group': {'gsm8k': gsm}, 'ifeval': ifeval}, 2048)
    assert '####' in gsm.config['doc_to_text']
    assert ifeval.config['doc_to_text'] == 'original'
    assert gsm.config['generation_kwargs']['until'] == ['STOP']
    assert ifeval.config['generation_kwargs']['max_gen_toks'] == 2048


def test_summary_deduplicates_filters_and_flags_sid(tmp_path):
    import json
    row = {'doc_id': 0, 'resps': [['<|sid_begin|><s_d_101>']],
           'filter': 'strict-match', 'filtered_resps': ['[invalid]']}
    other = dict(row, filter='flexible-extract', filtered_resps=['101'])
    (tmp_path / 'samples_gsm8k_2026-09-19T00-00-00.jsonl').write_text('\n'.join(map(json.dumps, [row, other])))
    report = protocol.summarize_samples(tmp_path)
    assert report['gsm8k']['samples'] == 1
    assert report['gsm8k']['sid'] == 1
    assert report['gsm8k']['invalid_by_filter']['strict-match'] == 1


def test_command_uses_protocol_launcher_and_auto_batch(tmp_path):
    from benchmarks.lm_eval.wrapper import build_command
    command = build_command('/models/qwen', {'tasks': ['gsm8k'], 'num_fewshot': 0, 'seed': 42}, tmp_path)
    assert command[2] == 'benchmarks.lm_eval.protocol'
    assert command[command.index('--batch_size') + 1] == 'auto'
    assert 'enable_thinking' not in command[command.index('--model_args') + 1]


def test_template_without_thinking_control_fails():
    import pytest
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: 'assistant\n')
    with pytest.raises(RuntimeError, match='non-thinking support'):
        protocol.render_chat(SimpleNamespace(tokenizer=tokenizer), [])


def test_rerun_replaces_only_owned_evaluation_outputs(tmp_path):
    from benchmarks.lm_eval.wrapper import prepare_output_dir
    results = tmp_path / 'results.json' / 'old_model'
    results.mkdir(parents=True)
    (results / 'samples_old.jsonl').write_text('old')
    (tmp_path / 'generation_diagnostics.jsonl').write_text('old')
    (tmp_path / 'manifest.json').write_text('old')
    (tmp_path / 'notes.txt').write_text('keep')
    prepare_output_dir(tmp_path)
    assert not (tmp_path / 'results.json').exists()
    assert not (tmp_path / 'generation_diagnostics.jsonl').exists()
    assert not (tmp_path / 'manifest.json').exists()
    assert (tmp_path / 'notes.txt').read_text() == 'keep'
    (tmp_path / 'results.json').write_text('{}')
    prepare_output_dir(tmp_path)
    assert not (tmp_path / 'results.json').exists()
