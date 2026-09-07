import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate
from pact_eval.worker import episode_counts


class ReportingTests(unittest.TestCase):
    def test_explicit_episode_outcomes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'internal_logs').mkdir()
            (out/'internal_logs/run.txt').write_text('Task: success\nSuccess: True\n# successes: 1\nSuccess: False\nEpisode error: broken\n')
            episodes,successes,errors=episode_counts(out)
            self.assertEqual((episodes,successes),(2,1))
            self.assertEqual(len(errors),1)

    def test_only_backend_matched_baseline(self):
        def record(method,kind,latency):
            return dict(id=method+kind,model='oft',suite='libero_goal',strategy=method,backend_kind=kind,
                        state='COMPLETED',policy_ms_mean=latency,model_ms_mean=latency/2)
        jobs=[record('vanilla','native',100),record('fastv','native',50),record('divprune','divprune',25)]
        with tempfile.TemporaryDirectory() as tmp:
            evaluate.write_summary(Path(tmp),jobs)
        self.assertEqual(jobs[1]['policy_speedup'],2)
        self.assertIsNone(jobs[2]['policy_speedup'])

    def test_verification_never_gets_formal_speedup(self):
        jobs=[dict(id='vanilla',model='oft',suite='libero_goal',strategy='vanilla',backend_kind='native',
                   state='VERIFIED',policy_ms_mean=100,model_ms_mean=80)]
        with tempfile.TemporaryDirectory() as tmp:
            evaluate.write_summary(Path(tmp),jobs)
        self.assertIsNone(jobs[0]['policy_speedup'])

    def test_path_safety(self):
        with self.assertRaises(ValueError):
            evaluate.run_path(ROOT)
        with self.assertRaises(ValueError):
            evaluate.run_path(ROOT/'runs')
        with self.assertRaises(ValueError):
            evaluate.run_path(ROOT/'runs/../../unrelated')

if __name__=='__main__':
    unittest.main()
