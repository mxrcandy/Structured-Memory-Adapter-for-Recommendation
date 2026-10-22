"""Exercise real training shell interfaces; mock GPU work, conversion and Slurm."""
import os, json, shutil, subprocess, tempfile, sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
py=sys.executable
profile=os.environ.get('TEST_DATA_PROFILE', 'amazon_letter_beauty')
exact_itm=os.environ.get('TEST_EXACT_ITM') == '1'
prior_itm=os.environ.get('TEST_PRIOR_ITM') == '1'
structured_itm=exact_itm or prior_itm
aux_weight=os.environ.get('TEST_ITM_AUX_WEIGHT')
with tempfile.TemporaryDirectory(prefix='amazon_pipeline_') as d:
    base=Path(d)
    scripts=base/'pretrain/scripts'
    shutil.copytree(root/'pretrain/scripts', scripts)
    (base/'process_data').symlink_to(root/'process_data', target_is_directory=True)
    (base/'hf_models').symlink_to(root/'hf_models', target_is_directory=True)
    (base/'benchmarks').symlink_to(root/'benchmarks', target_is_directory=True)
    bin=base/'bin'; bin.mkdir()
    mock='''#!%s
import sys,os,json,shutil
from pathlib import Path
a=sys.argv[1:]
def arg(k): return a[a.index(k)+1]
with open(os.environ['TRACE'],'a') as f: f.write(json.dumps([Path(sys.argv[0]).name,a])+ '\\n')
if Path(sys.argv[0]).name=='mpirun':
    if a == ['--version']:
        print('mpirun (Open MPI) 4.1.5')
        sys.exit(0)
    assert arg('-np')=='1'
    assert arg('--max_length')=='2048'
    config=json.loads(Path(arg('--dataset_config')).read_text())
    assert config['itemic_id_range']==[151669,152694]
    assert config['cut_to_pad']==1
    assert all(Path(p).is_file() for p in json.loads(Path(config['sources']).read_text()))
    out=Path(arg('--output_dir')); (out/'step7/global_step7').mkdir(parents=True)
    (out/'latest').write_text('global_step7')
elif a and a[0].endswith('convert_checkpoint_to_hf.py'):
    assert Path(arg('--checkpoint_dir')).is_dir()
    out=Path(arg('--output_dir')); out.mkdir()
    shutil.copy(Path(arg('--source_hf_model_path'))/'config.json',out/'config.json')
elif Path(sys.argv[0]).name=='sbatch':
    assert '--dependency=afterok:1234' in a and not any(x.startswith('--gres=gpu:') for x in a)
    assert a[-1].endswith('slurm_eval_amazon.sbatch')
    exports=next(v for v in a if v.startswith('--export='))
    task='amazon_' + os.environ['DATA_PROFILE'].removeprefix('amazon_letter_').split('_general')[0]
    assert exports == '--export=ALL'
    assert os.environ['TASK_TYPES']==task
    print('Submitted batch job 4242')
else: os.execv(%r,[%r]+a)
''' % (py,py,py)
    for name in ['python3','mpirun','sbatch']:
        p=bin/name;p.write_text(mock);p.chmod(0o755)
    env={'PATH':str(bin)+':'+str(Path(py).parent)+':/usr/bin:/bin', 'HOME':os.environ['HOME'],
         'DATA_PROFILE':profile,'NUM_GPUS':'1','RUN_ID':'smoke','SLURM_JOB_ID':'1234',
         'RUN_GENERAL_EVAL':'false','TRACE':str(base/'trace'), 'MODEL_DIR':str(base/'hf_models/Qwen3-0.6B_itemic')}
    def run(cmd):
        r=subprocess.run(cmd,cwd=base/'pretrain',env=env,capture_output=True,text=True,timeout=90)
        assert r.returncode==0, r.stdout+'\n'+r.stderr
    run(['bash',str(scripts/'train_stage1_itemic.sh')])
    stage1=base/f'model_output/stage1_itemic_alignment_{profile}_smoke'
    run(['bash',str(scripts/'convert_checkpoint_to_hf.sh'),env['MODEL_DIR'],str(stage1),'7'])
    del env['MODEL_DIR']
    env['STAGE1_CONVERTED_MODEL_DIR']=str(stage1/'step7/global_step7/converted')
    if structured_itm:
        pattern='ab_to_a_prior_min5.json' if prior_itm else 'ab_to_ab_exact_min5.json'
        env.update(ADAPTER_ITM_MODE='successor_prior' if prior_itm else 'exact', E2E_RUN_TAG='exact_smoke',
                   ADAPTER_ITM_PATTERN_FILE=str(root/'raw_data/letter/Beauty/itm_patterns'/pattern))
        if aux_weight is not None:
            env['ADAPTER_ITM_AUX_LOSS_WEIGHT'] = aux_weight
    run(['bash',str(scripts/('train_e2e_adapter_structure.sh' if structured_itm else 'train_e2e_full.sh'))])
    events=[json.loads(l) for l in (base/'trace').read_text().splitlines()]
    trains=[a for n,a in events if n=='mpirun' and '-np' in a]
    conversions=[a for n,a in events if a and a[0].endswith('convert_checkpoint_to_hf.py')]
    assert len(trains)==3 and len(conversions)==3, [(n, a[:3]) for n,a in events]
    if structured_itm:
        for train in trains[1:]:
            assert train[train.index('--adapter_itm_mode')+1]==env['ADAPTER_ITM_MODE']
            assert train[train.index('--adapter_itm_pattern_file')+1]==env['ADAPTER_ITM_PATTERN_FILE']
            assert train[train.index('--adapter_itm_order')+1]=='1'
            assert train[train.index('--adapter_itm_pattern_min_frequency')+1]=='5'
            if aux_weight is not None:
                assert train[train.index('--adapter_itm_aux_loss_weight')+1] == aux_weight
    prev=str(base/'hf_models/Qwen3-0.6B_itemic')
    for train,conv in zip(trains,conversions):
        assert train[train.index('--model_dir')+1]==prev
        assert conv[conv.index('--source_hf_model_path')+1]==prev
        prev=conv[conv.index('--output_dir')+1]
    tag='exact_smoke' if structured_itm else 'full'
    sft=base/f'model_output/e2e_sft_{profile}_{tag}_smoke'
    assert Path(prev)==sft/'step7/global_step7/converted'
    summary=(sft/'end_to_end_summary.log').read_text()
    assert 'eval_job_id: 4242' in summary
    print('PASS: real Stage1/config generation -> mocked conversion -> real Stage2/SFT/config generation -> mocked conversions -> dependent Amazon eval submission. All three model handoffs, shard paths, SID range, max length, single-GPU commands and SFT naming checked. GPU compute and weight conversion mocked.')
