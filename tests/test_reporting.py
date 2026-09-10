import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate
from pact_eval.table1 import table_rows
from pact_eval.worker import episode_counts, expected_native_budgets


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

    def test_adaptive_pact_has_no_fake_fixed_ratio(self):
        jobs=[]
        for suite in ('spatial','object','goal','long'):
            jobs.append(dict(model='oft',strategy='pact-vla',backend_kind='native',ratio=0.0,
                ratio_semantics='adaptive_retention',suite_key=suite,state='COMPLETED',mode='eval',
                success_rate=.9,policy_ms_mean=100,model_ms_mean=80))
        row=table_rows(jobs)[0]
        self.assertEqual(row['method'],'pact-vla')
        self.assertIsNone(row['pruning_or_reuse_pct'])
        self.assertIsNone(row['retention_pct'])

    def test_adaptive_pact_accepts_each_candidate_budget(self):
        job=dict(model='oft',strategy='pact-vla',ratio=0.0,
                 ratio_semantics='adaptive_retention',
                 settings={'pact_budget_rates':'0.125,0.25,0.5,0.75,1'})
        self.assertEqual(expected_native_budgets(job,512),{64,128,256,384,512})

if __name__=='__main__':
    unittest.main()
