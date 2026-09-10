import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate
from pact_eval.planning import make_jobs, pact_budget_rates, ratio, suite_keys


class PlanningTests(unittest.TestCase):
    def jobs(self, command):
        return make_jobs(ROOT,evaluate.parser().parse_args(command.split()))

    def test_matrix(self):
        jobs=self.jobs('--model all --strategy all --suite all --ratio 50 75 87.5')
        self.assertEqual(len(jobs),132)
        self.assertEqual(sum(j['expected_episodes'] for j in jobs),1320)
        self.assertEqual({j['suite'] for j in jobs},{'libero_spatial','libero_object','libero_goal','libero_10'})

    def test_matched_baselines(self):
        jobs=self.jobs('--model all --strategy all --suite all --ratio 50 75 87.5 --with-baseline')
        self.assertEqual(len(jobs),148)
        for kind in ('native','divprune','vla-cache'):
            self.assertEqual(sum(j['strategy']=='vanilla' and j['backend_kind']==kind for j in jobs),8)

    def test_native_configs(self):
        jobs=self.jobs('--model all --strategy fastv sparsevlm vla-pruner --task-ids 9 2 9 --trials 5')
        for j in jobs:
            self.assertEqual(j['task_ids'],[9,2])
            self.assertEqual(j['expected_episodes'],10)
            self.assertFalse(j['settings']['use_prefil_attention'])
            if j['model']=='openvla':
                self.assertFalse(j['settings']['use_cache'])
                self.assertEqual(j['settings']['use_temporal'],j['strategy']=='vla-pruner')
            else:
                self.assertFalse(j['settings']['use_vla_cache'])

    def test_backends(self):
        jobs=self.jobs('--model all --strategy vla-cache divprune --ratio 87.5 --with-baseline')
        self.assertEqual(len(jobs),8)
        for j in jobs:
            if j['backend_kind']=='vla-cache':
                self.assertIsNone(j['transformers_override'])
                self.assertIn('vla-cache',j['environment'])

    def test_pact_vla_is_one_adaptive_oft_condition(self):
        jobs=self.jobs('--model oft --strategy pact-vla --suite spatial --ratio 25 50 75')
        self.assertEqual(len(jobs),1)
        job=jobs[0]
        self.assertEqual(job['id'],'oft_spatial_pact-vla_native_adaptive')
        self.assertEqual(job['ratio_semantics'],'adaptive_retention')
        self.assertEqual(job['ratio'],0.0)
        self.assertTrue(job['settings']['use_pact_vla'])
        self.assertFalse(job['settings']['use_vla_pruner'])
        self.assertFalse(job['settings']['use_fastv'])
        self.assertTrue(job['settings']['use_prefil_attention'])
        self.assertEqual(job['settings']['fastv_r'],.5)
        self.assertEqual(job['settings']['pact_budget_rates'],'0.125,0.25,0.5,0.75,1')

    def test_pact_vla_model_compatibility(self):
        with self.assertRaisesRegex(ValueError,'only for --model oft'):
            self.jobs('--model openvla --strategy pact-vla')
        jobs=self.jobs('--model all --strategy pact-vla --suite all')
        self.assertEqual(len(jobs),4)
        self.assertEqual({job['model'] for job in jobs},{'oft'})

    def test_invalid(self):
        for command in ('--task-ids 10','--trials 0','--ratio 100','--ratio nan','--ratio 0',
                        '--strategy divprune --model openvla --decode-cache on',
                        '--strategy vla-cache --decode-cache off',
                        '--backend-option fastv_r 0.99','--backend-option pact_budget_rates 0.5,1',
                        '--pact-budget-rates 0.5,0.25,1','--pact-budget-rates 0.25,0.5',
                        '--model all --checkpoint fake'):
            with self.subTest(command=command),self.assertRaises(ValueError):
                self.jobs(command)

    def test_aliases(self):
        self.assertEqual(ratio('87.5%'),.875)
        self.assertEqual(ratio('.5'),.5)
        self.assertEqual(pact_budget_rates('0.125, .5, 1.0'),'0.125,0.5,1')
        self.assertEqual(suite_keys(['libero_10','libero_spatial']),['long','spatial'])

if __name__=='__main__':
    unittest.main()
