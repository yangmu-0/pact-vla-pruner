from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate
from pact_eval.metrics import layer_cost,prefill_cost,should_profile,summarize_calls
from pact_eval.planning import make_jobs
from pact_eval.table1 import ratio_percent,table_rows


class MetricTests(unittest.TestCase):
    def test_paper_ffn_and_gated_are_distinct(self):
        self.assertEqual(layer_cost(1,10,4,8,3)-layer_cost(1,10,4,8,2),10*4*8)
        visits={i:[(1,10,4,8)]*7 for i in range(32)}
        data=prefill_cost(visits)
        self.assertEqual(data['llm_forward_count'],7)
        self.assertAlmostEqual(data['llm_prefill_eq9_T'],32*layer_cost(1,10,4,8,2)/1e12)
        # Prefill metric must not silently multiply by seven no-cache forwards.
        self.assertEqual(data['llm_layer_visit_counts'],[7]*32)

    def test_sampler(self):
        self.assertEqual([i for i in range(1,102) if should_profile(i,50)],[1,5,50,100])

    def test_relative_acc_formula_and_published_mismatch(self):
        # Paper reports 97.43, not reconstructable from the displayed rates by
        # this explicit formula. Test the arithmetic, never force-fit the table.
        value=ratio_percent([86.2,81.6,77.2,50.6],[87.6,84.6,78.6,52.2])
        self.assertAlmostEqual(value,97.50235565254017)
        self.assertNotEqual(round(value,2),97.43)
        self.assertIsNone(ratio_percent([1,1,1,1],[1,1,0,1]))
        self.assertIsNone(ratio_percent([1],[1]))

    def test_four_rates_and_baselines(self):
        args=evaluate.parser().parse_args('--model all --strategy all --suite all --ratio 25 50 75 87.5 --trials 3 --with-baseline --collect-flops --condition-order method'.split())
        jobs=make_jobs(ROOT,args)
        self.assertEqual(len(jobs),188)
        self.assertEqual(sum(j['expected_episodes'] for j in jobs),5640)
        self.assertTrue(all(j['strategy']=='vanilla' for j in jobs[:24]))
        self.assertEqual(len(table_rows(jobs)),43)
        self.assertTrue(all(r['Acc.(%)'] is None for r in table_rows(jobs)))

    def test_table_macro_average_and_partial_guard(self):
        jobs=[]
        for s in ('spatial','object','goal','long'):
            for method,rate in [('vanilla',.5),('fastv',.25)]:
                jobs.append(dict(model='oft',strategy=method,backend_kind='native',ratio=0 if method=='vanilla' else .25,
                    suite_key=s,mode='eval',state='COMPLETED',success_rate=rate,policy_ms_mean=10,model_ms_mean=8,
                    flops_eq9_prefill_T=2,flops_gated_prefill_T=3,flops_profiled_call_T=7,flops_profiled_warm_T=7))
        rows=table_rows(jobs)
        self.assertEqual(rows[1]['Acc.(%)'],50)
        self.assertEqual(rows[1]['Success_avg(%)'],25)
        jobs[-1]['state']='RUNNING'
        self.assertIsNone(table_rows(jobs)[1]['Acc.(%)'])
        self.assertIsNone(table_rows(jobs)[1]['Latency(ms)'])

    def test_profile_summary(self):
        calls=[dict(profiled_flops=1,profiled_call_flops_T=10,task_id=3,episode_call=1,llm_prefill_eq9_T=1,llm_prefill_gated_T=2,llm_forward_count=7),
               dict(profiled_flops=1,profiled_call_flops_T=6,task_id=3,episode_call=5,llm_prefill_eq9_T=.5,llm_prefill_gated_T=1,llm_forward_count=7)]
        result=summarize_calls(calls)
        self.assertEqual(result['flops_profiled_call_T'],8)
        self.assertEqual(result['flops_profiled_warm_T'],6)
        self.assertEqual(result['flops_eq9_prefill_T'],.75)

if __name__=='__main__':
    unittest.main()
