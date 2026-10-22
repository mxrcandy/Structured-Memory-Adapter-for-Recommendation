"""Exercise E2E Slurm submission with a fake scheduler; never launch jobs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EvalSubmissionTest(unittest.TestCase):
    def run_submission(self, profile, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / 'capture.json'
            scheduler = Path(directory) / 'sbatch'
            scheduler.write_text(
                '#!/usr/bin/env python3\nimport os,sys,json\n'
                'from pathlib import Path\n'
                'Path(os.environ["CAPTURE"]).write_text(json.dumps([sys.argv[1:],dict(os.environ)]))\n'
                'print("Submitted batch job 4242")\n'
            )
            scheduler.chmod(0o755)
            env = {
                'PATH': os.environ['PATH'], 'HOME': os.environ['HOME'],
                'DATA_PROFILE': profile, 'SLURM_JOB_ID': '1234', 'NUM_GPUS': '1',
                'E2E_SBATCH_BIN': str(scheduler), 'CAPTURE': str(capture), **overrides,
            }
            result = subprocess.run(
                ['bash', '-c', 'source pretrain/scripts/e2e_rec_eval.sh; submit_e2e_rec_eval /tmp/model'],
                cwd=ROOT, env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(capture.read_text()) if capture.exists() else ([], {})

    def test_amazon_route_leaves_gpu_request_to_sbatch_file(self):
        args, env = self.run_submission('amazon_letter_beauty', REC_EVAL_SAMPLE_SIZE='3000')
        self.assertEqual(args[-1], str(ROOT / 'benchmarks/slurm_eval_amazon.sbatch'))
        self.assertIn('--dependency=afterok:1234', args)
        self.assertIn('--export=ALL', args)
        self.assertFalse(any(arg.startswith('--gres=gpu:') for arg in args))
        self.assertEqual(env['NUM_GPUS'], '1')
        self.assertEqual(env['TASK_TYPES'], 'amazon_beauty')
        self.assertEqual(env['SAMPLE_SIZE'], '3000')
        self.assertEqual(env['MODEL_PATH'], '/tmp/model')

    def test_video_route_and_task_type(self):
        args, env = self.run_submission('10pct')
        self.assertEqual(args[-1], str(ROOT / 'benchmarks/slurm_eval_video.sbatch'))
        self.assertEqual(env['TASK_TYPES'], 'video')
        self.assertFalse(any(arg.startswith('--gres=gpu:') for arg in args))

    def test_instruments_and_other_amazon_routes(self):
        for profile, task in (
            ('amazon_letter_instruments', 'amazon_instruments'),
            ('amazon_letter_instruments_general1_rec16', 'amazon_instruments'),
            ('amazon_letter_toys', 'amazon_toys'),
            ('amazon_letter_sports', 'amazon_sports'),
            ('amazon_letter_yelp', 'amazon_yelp'),
        ):
            with self.subTest(profile=profile):
                args, env = self.run_submission(profile)
                self.assertEqual(args[-1], str(ROOT / 'benchmarks/slurm_eval_amazon.sbatch'))
                self.assertEqual(env['TASK_TYPES'], task)

    def test_explicit_script_override(self):
        script = str(ROOT / 'benchmarks/slurm_eval_video.sbatch')
        args, _ = self.run_submission('amazon_letter_beauty', E2E_EVAL_SBATCH_SCRIPT=script)
        self.assertEqual(args[-1], script)

    def test_disabled_does_not_submit(self):
        args, _ = self.run_submission('amazon_letter_beauty', RUN_REC_EVAL='false')
        self.assertEqual(args, [])


if __name__ == '__main__':
    unittest.main()
