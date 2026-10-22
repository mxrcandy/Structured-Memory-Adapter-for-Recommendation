"""Check scheduler arguments and child environments without submitting jobs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EvalEnvCompatTest(unittest.TestCase):
    def submit(self, helper, mode, **extra):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / 'capture.json'
            scheduler = Path(directory) / 'sbatch'
            scheduler.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                                 'with open(os.environ["CAPTURE"],"w") as f: json.dump([sys.argv[1:],dict(os.environ)],f)\n'
                                 'print("Submitted batch job 4242")\n')
            scheduler.chmod(0o755)
            env = dict(PATH=os.environ['PATH'], HOME=os.environ['HOME'],
                       CAPTURE=str(capture), E2E_SBATCH_BIN=str(scheduler),
                       SLURM_JOB_ID='1234', RUN_GENERAL_EVAL='true',
                       DATA_PROFILE='amazon_letter_instruments', NUM_GPUS='1',
                       CONDA_ROOT='/opt/Software/Anaconda3',
                       REC_EVAL_SAMPLE_SIZE='3000', WORKER_BATCH_SIZE='128',
                       REC_EVAL_RESULT_NAME='rec result', GENERAL_EVAL_RESULT_NAME='general result',
                       INHERITED_MARKER='retained', **extra)
            if mode:
                env['E2E_EVAL_ENV_MODE'] = mode
            command = (f'source pretrain/scripts/e2e_{helper}_eval.sh; '
                       f'submit_e2e_{helper}_eval "/tmp/model with spaces"')
            result = subprocess.run(['bash', '-eu', '-c', command], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(capture.read_text()) if capture.exists() else None

    def test_all_modes_use_plain_all_and_preserve_parameters(self):
        for helper in ('rec', 'general'):
            with self.subTest(helper=helper):
                old_args, _ = self.submit(helper, None)
                new_args, new_env = self.submit(helper, 'process')
                self.assertIn('--export=ALL', old_args)
                self.assertIn('--export=ALL', new_args)
                self.assertEqual([a for a in old_args if not a.startswith('--export=')],
                                 [a for a in new_args if not a.startswith('--export=')])
                for key, value in {'MODEL_PATH': '/tmp/model with spaces',
                                   'CONDA_ROOT': '/opt/Software/Anaconda3',
                                   'ADAPTER_ENABLED': 'true',
                                   'RESULT_NAME': 'rec result' if helper == 'rec' else 'general result'}.items():
                    self.assertEqual(new_env[key], value)
                self.assertEqual(new_env['INHERITED_MARKER'], 'retained')
                self.assertIn('--dependency=afterok:1234', new_args)
                if helper == 'rec':
                    self.assertEqual(new_env['TASK_TYPES'], 'amazon_instruments')

    def test_disabled_does_not_submit(self):
        self.assertIsNone(self.submit('rec', 'process', RUN_REC_EVAL='false'))

    def test_process_preserves_commas(self):
        _, env = self.submit('rec', 'process', REC_EVAL_ADAPTER_ENABLED='true',
                             BENCHMARK_CONDA_ENV='/tmp/env,with spaces')
        self.assertEqual(env['BENCHMARK_CONDA_ENV'], '/tmp/env,with spaces')


if __name__ == '__main__':
    unittest.main()
