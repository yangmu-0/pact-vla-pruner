"""Table-1 shaped exports with explicit canonical and matched baselines."""
import csv
import json
from pathlib import Path
import statistics
from .metrics import METRIC_NOTES

SUITES=('spatial','object','goal','long')


def ratio_percent(values, baselines):
    if len(values)!=4 or len(baselines)!=4 or any(v is None for v in values) or any(b is None or b<=0 for b in baselines):
        return None
    return 100*statistics.mean(v/b for v,b in zip(values,baselines))


def table_rows(jobs, supplemental=False):
    groups={}
    for job in jobs:
        if job.get('mode','eval')!='eval':
            continue
        key=(job['model'],job['strategy'],job['backend_kind'],job['ratio'])
        groups.setdefault(key,{})[job['suite_key']]=job
    def complete(group):
        return len(group)==4 and all(group[s]['state']=='COMPLETED' for s in SUITES)
    def metric(group,field):
        values=[group[s].get(field) for s in SUITES] if complete(group) else []
        return statistics.mean(values) if len(values)==4 and all(v is not None for v in values) else None
    output=[]
    for (model,strategy,kind,pruning),group in groups.items():
        if strategy=='vanilla' and kind!='native' and not supplemental:
            continue
        canonical=groups.get((model,'vanilla','native',0.0),{})
        matched=groups.get((model,'vanilla',kind,0.0),{})
        current=[group[s].get('success_rate') for s in SUITES] if complete(group) else []
        base=[canonical[s].get('success_rate') for s in SUITES] if complete(canonical) else []
        match=[matched[s].get('success_rate') for s in SUITES] if complete(matched) else []
        row={'model':model,'method':strategy,'backend':kind,'pruning_or_reuse_pct':pruning*100,
             'retention_pct':100*(1-pruning),'completed_suites':sum(j['state']=='COMPLETED' for j in group.values()),
             'state':'COMPLETE' if complete(group) else 'PARTIAL',
             **{s:100*group[s]['success_rate'] if s in group and group[s]['state']=='COMPLETED' and group[s].get('success_rate') is not None else None for s in SUITES},
             'Success_avg(%)':100*statistics.mean(current) if len(current)==4 and all(v is not None for v in current) else None,
             'Acc.(%)':ratio_percent(current,base),'Acc_matched(%)':ratio_percent(current,match),
             'Acc_ratio_of_means(%)':100*sum(current)/sum(base) if len(current)==len(base)==4 and all(v is not None for v in current+base) and sum(base)>0 else None,
             'FLOPs(T)':metric(group,'flops_gated_prefill_T'),
             'FLOPs_paper_Eq9(T)':metric(group,'flops_eq9_prefill_T'),
             'FLOPs_profiled_call(T)':metric(group,'flops_profiled_call_T'),
             'FLOPs_profiled_warm_call(T)':metric(group,'flops_profiled_warm_T'),
             'Latency(ms)':metric(group,'policy_ms_mean'),'Model_latency(ms)':metric(group,'model_ms_mean')}
        latency=row['Latency(ms)']
        base_latency=metric(canonical,'policy_ms_mean')
        matched_latency=metric(matched,'policy_ms_mean')
        row['Speedup(x)']=base_latency/latency if base_latency is not None and latency else None
        row['Matched_speedup(x)']=matched_latency/latency if matched_latency is not None and latency else None
        base_flops=metric(canonical,'flops_gated_prefill_T')
        row['FLOPs_ratio(%)']=100*row['FLOPs(T)']/base_flops if base_flops and row['FLOPs(T)'] is not None else None
        output.append(row)
    return output


def write_table1(path,jobs):
    path=Path(path)
    for name,supplemental in [('table1.csv',False),('table1_with_backend_baselines.csv',True)]:
        rows=table_rows(jobs,supplemental)
        if not rows:
            continue
        temp=path/(name+'.tmp')
        with temp.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temp.replace(path/name)
    (path/'metric_definitions.json').write_text(json.dumps(METRIC_NOTES,indent=2))
