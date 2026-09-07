#!/usr/bin/env python3
"""Read-only consistency audit of a completed metrics-enabled unified run."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pact_eval.metrics import should_profile
from pact_eval.worker import episode_counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    path=args.run.resolve()
    if not path.is_relative_to(ROOT/'runs') or path==ROOT/'runs':
        raise ValueError('Expected a concrete run under this repository runs/.')
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['state']=='COMPLETE',manifest['state']
    checks=[]
    for job in manifest['jobs']:
        assert job['collect_flops']
        out=Path(job['output'])
        report=json.loads((out/'result.json').read_text())
        calls=list(csv.DictReader((out/'policy_timing_calls.csv').open()))
        profiles=[json.loads(line) for line in (out/'flops_profile_samples.jsonl').read_text().splitlines()]
        expected_profiled=[c for c in calls if should_profile(int(c['episode_call']),job['flops_sample_interval'])]
        assert len(calls)==report['successful_calls']==report['flops_formula_calls']
        assert len(profiles)==len(expected_profiled)==report['flops_profiled_samples']
        assert [p['call'] for p in profiles]==[int(c['call']) for c in expected_profiled]
        for call in calls:
            profiled=should_profile(int(call['episode_call']),job['flops_sample_interval'])
            assert int(call['profiled_flops'])==int(profiled)
            assert int(call['warm_sample'])==int(int(call['call'])>job['warmup_calls'] and not profiled)
            assert float(call['llm_prefill_gated_T'])>float(call['llm_prefill_eq9_T'])>0
        for sample,call in zip(profiles,expected_profiled):
            assert sum(sample['registered_operation_flops'].values())==sample['total_flops']
            assert sample['total_flops']>=2*float(call['llm_prefill_eq9_T'])*1e12
            assert math.isclose(sample['total_flops']/1e12,float(call['profiled_call_flops_T']))
        eligible=[c for c in calls if int(c['warm_sample'])]
        assert len(eligible)==report['measured_calls']
        for field,csv_field in [('policy_ms_mean','policy_ms'),('model_ms_mean','model_ms')]:
            expected=statistics.mean(float(c[csv_field]) for c in eligible) if eligible else None
            assert (expected is None and report[field] is None) or math.isclose(expected,report[field],rel_tol=1e-9)
        for field,csv_field in [('flops_eq9_prefill_T','llm_prefill_eq9_T'),('flops_gated_prefill_T','llm_prefill_gated_T')]:
            assert math.isclose(report[field],statistics.mean(float(c[csv_field]) for c in calls),rel_tol=1e-9)
        episodes,successes,errors=episode_counts(out)
        assert not errors and not report['policy_errors'] and not report['episode_errors']
        assert (episodes,successes)==(report['episodes'],report['successes'])
        if job['mode']=='eval':
            assert report['state']=='COMPLETED' and episodes==job['expected_episodes']
            assert set(report['flops_profiled_task_ids'])==set(job['task_ids'])
            assert len({(c['task_id'],c['episode']) for c in calls})==episodes
        else:
            assert report['state']=='VERIFIED' and len(calls)==job['verify_calls']
            assert report['success_rate'] is None
        checks.append(dict(id=job['id'],state=report['state'],episodes=episodes,successes=successes,
                           policy_calls=len(calls),profiled_calls=len(profiles),latency_calls=len(eligible)))
    print(json.dumps(dict(state='PASS',run=str(path),conditions=len(checks),checks=checks),indent=2))


if __name__=='__main__':
    main()
