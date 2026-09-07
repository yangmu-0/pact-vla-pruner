import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate
from pact_eval.planning import make_jobs, ratio, suite_keys


class PlanningTests(unittest.TestCase):
    def jobs(self, command):
        return make_jobs(ROOT,evaluate.parser().parse_args(command.split()))

    def test_matrix(self):
        jobs=self.jobs('--model all --strategy all --suite all --ratio 50 75 87.5')
        self.assertEqual(len(jobs),128)
        self.assertEqual(sum(j['expected_episodes'] for j in jobs),1280)
        self.assertEqual({j['suite'] for j in jobs},{'libero_spatial','libero_object','libero_goal','libero_10'})

    def test_matched_baselines(self):
        jobs=self.jobs('--model all --strategy all --suite all --ratio 50 75 87.5 --with-baseline')
        self.assertEqual(len(jobs),144)
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

    def test_invalid(self):
        for command in ('--task-ids 10','--trials 0','--ratio 100','--ratio nan','--ratio 0',
                        '--strategy divprune --model openvla --decode-cache on',
                        '--strategy vla-cache --decode-cache off',
                        '--backend-option fastv_r 0.99','--model all --checkpoint fake'):
            with self.subTest(command=command),self.assertRaises(ValueError):
                self.jobs(command)

    def test_aliases(self):
        self.assertEqual(ratio('87.5%'),.875)
        self.assertEqual(ratio('.5'),.5)
        self.assertEqual(suite_keys(['libero_10','libero_spatial']),['long','spatial'])

if __name__=='__main__':
    unittest.main()
