#!/usr/bin/env python3
"""Single entrypoint for archived VLA implementations and new, isolated runs."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import subprocess
import sys
import uuid

from pact_eval.planning import (MODELS, STRATEGIES, SUITES, backend, environment,
                               make_jobs, suite_keys, validate_job)
from pact_eval.table1 import write_table1

ROOT = Path(__file__).resolve().parent

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def save(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def boolean(value):
    if value.lower() not in ('true','false'):
        raise argparse.ArgumentTypeError('Use true or false.')
    return value.lower() == 'true'

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', nargs='+', default=['oft'], help='openvla, oft, or all')
    p.add_argument('--strategy', nargs='+', default=['vanilla'], help=', '.join(STRATEGIES)+', or all')
    p.add_argument('--suite', nargs='+', default=['spatial'], help='spatial, object, goal, long, or all; libero_* aliases accepted')
    p.add_argument('--ratio','--prune-ratio', nargs='+', default=['0.5'], help='Removed fraction; VLA-Cache: target reused fraction. Decimal or percent.')
    p.add_argument('--task-ids', nargs='+', type=int, help='Zero-based task IDs 0..9; omitted means all 10 tasks')
    p.add_argument('--trials', type=int, default=1, help='Episodes per selected task, using initial states 0..trials-1')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--checkpoint', help='Override only when selecting one model and one suite')
    p.add_argument('--prune-layer', type=int, default=3, help='FastV/SparseVLM/VLA-Pruner fastv_k; DivPrune is pre-LLM')
    p.add_argument('--prefill-attention', type=boolean, default=False)
    p.add_argument('--decode-cache', choices=['auto','on','off'], default='auto', help='auto: off for native OpenVLA/DivPrune; official cached path for VLA-Cache')
    p.add_argument('--with-baseline', action='store_true', help='Add matching native/cache/DivPrune Vanilla backend for each strategy')
    p.add_argument('--baseline-backend', choices=['native','divprune','vla-cache'], default='native')
    p.add_argument('--backend-option', nargs=2, action='append', default=[], metavar=('FIELD','VALUE'), help='Advanced backend dataclass option; may not override method/task/count identity')
    p.add_argument('--mode', choices=['eval','verify'], default='eval', help='verify stops after N valid policy calls; it is NOT a success-rate trial')
    p.add_argument('--verify-calls', type=int, default=5, help='At least 5 to cross the OFT temporal warmup')
    p.add_argument('--warmup-calls', type=int, default=10, help='Exclude first N calls from reported latency, not from success counts')
    p.add_argument('--collect-flops',action='store_true',help='Record paper Eq.9, gated-MLP formula and separate sampled registered-operation FLOPs')
    p.add_argument('--flops-sample-interval',type=int,default=50,help='Profile calls 1,5 of each episode and every N calls; 0 disables periodic samples. Profiled calls excluded from latency.')
    p.add_argument('--condition-order',choices=['suite','method'],default='suite',help='method runs baselines first, then groups ratios across models/suites')
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--name', help='Safe run directory name under runs/; never overwrites')
    p.add_argument('--dry-run', action='store_true', help='Validate resources and print the exact matrix without creating runs or using GPU')
    p.add_argument('--tmux', action='store_true', help='Run detached in tmux; every condition remains sequential')
    p.add_argument('--resume', type=Path, help='Resume only this new unified run; completed conditions are skipped')
    p.add_argument('--retry-failed', action='store_true', help='On resume, use a new attempt directory for failed/interrupted conditions')
    p.add_argument('--status', type=Path, help='Read progress for an existing unified run')
    p.add_argument('--pause', type=Path, help='Request a pause at the next condition boundary')
    p.add_argument('--list', action='store_true', help='List choices and backend routing')
    p.add_argument('--list-tasks', action='store_true', help='List exact LIBERO task names without loading a model')
    p.add_argument('--doctor', action='store_true', help='Check all four Python environments and the eight checkpoint families')
    return p

def run_path(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT/'runs') or path == ROOT/'runs':
        raise ValueError('Only a concrete run inside this repository runs/ is accepted.')
    return path

def status(path):
    path = run_path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    for job in manifest['jobs']:
        if job['state']=='RUNNING' and job.get('output'):
            live = Path(job['output'])/'result.json'
            if live.exists():
                progress = json.loads(live.read_text())
                job.update({k:progress.get(k) for k in ('episodes','successes','successful_calls')})
        print(f"{job['state']:12} {job['id']} episodes={job.get('episodes',0)}/{job['expected_episodes']} successes={job.get('successes','-')}")
        if job.get('successful_calls') is not None:
            print('  policy calls:', job['successful_calls'])
    print('Run:', path, 'state:', manifest['state'])
    if (path/'PAUSE_REQUEST').exists():
        print('Pause requested; a running condition is allowed to finish.')

def nvidia_pids(gpu):
    p = subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],
                       text=True, capture_output=True, check=True)
    return p.stdout.strip()

def write_summary(path, jobs):
    baselines = {(j['model'],j['suite'],j['backend_kind']):j for j in jobs
                 if j['strategy']=='vanilla' and j['state']=='COMPLETED'}
    for job in jobs:
        base = baselines.get((job['model'],job['suite'],job['backend_kind']))
        for metric in ('policy','model'):
            value = job.get(metric+'_ms_mean')
            base_value = base.get(metric+'_ms_mean') if base else None
            job[metric+'_speedup'] = base_value/value if value and base_value and job['state']=='COMPLETED' else None
    fields = ['id','model','strategy','backend_kind','suite','ratio','ratio_semantics','mode','state',
              'episodes','successes','success_rate','policy_ms_mean','model_ms_mean','policy_speedup','model_speedup',
              'flops_eq9_prefill_T','flops_gated_prefill_T','flops_profiled_call_T','flops_profiled_warm_T',
              'flops_profiled_samples','flops_formula_calls','latency_profiled_calls_excluded','measured_calls','attempt']
    with (path/'summary.csv.tmp').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(jobs)
    (path/'summary.csv.tmp').replace(path/'summary.csv')
    if any(j.get('collect_flops') for j in jobs):
        write_table1(path,jobs)

def execute(path, manifest, retry):
    import fcntl
    locks = ROOT/'.locks'
    locks.mkdir(exist_ok=True)
    gpu = manifest['gpu']
    with (locks/f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Reload after locking: a concurrent runner may have finished since the
        # caller read the manifest. Never schedule from a stale in-memory copy.
        manifest = json.loads((path/'manifest.json').read_text())
        if (path/'PAUSE_REQUEST').exists():
            marker = path/'PAUSE_REQUEST'
            marker.rename(path/('PAUSE_REQUEST.resumed_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]))
        manifest.update(state='RUNNING', runner_pid=os.getpid())
        save(path/'manifest.json', manifest)
        for job in manifest['jobs']:
            if job['state'] in ('COMPLETED','VERIFIED'):
                continue
            if job['state'] in ('FAILED','INTERRUPTED','RUNNING') and not retry:
                raise ValueError('Incomplete condition requires --retry-failed; no previous results will be overwritten.')
            if (path/'PAUSE_REQUEST').exists():
                manifest['state'] = 'PAUSED'
                save(path/'manifest.json', manifest)
                print('PAUSED at condition boundary:', path, flush=True)
                return
            occupied = nvidia_pids(gpu)
            if occupied:
                manifest.update(state='BLOCKED_GPU_BUSY', occupied_pids=occupied)
                save(path/'manifest.json', manifest)
                raise ValueError(f'GPU {gpu} already has process(es): {occupied}. Nothing was stopped.')
            try:
                validate_job(ROOT, job)
            except Exception as error:
                manifest.update(state='FAILED_PREFLIGHT', error=str(error))
                save(path/'manifest.json', manifest)
                raise
            attempt = job.get('attempt', 0)+1
            out = path/job['id']/f'attempt_{attempt:02d}'
            out.mkdir(parents=True, exist_ok=False)
            (out/'internal_logs').mkdir()
            payload = {**job, 'output':str(out), 'root':str(ROOT), 'gpu':gpu,
                       'attempt':attempt, 'save_video':manifest['save_video']}
            save(out/'job.json', payload)
            command = [job['python'], manifest.get('worker_snapshot',str(ROOT/'pact_eval/worker.py')), str(out/'job.json')]
            job.update(state='RUNNING', attempt=attempt, output=str(out), started_at=now(), command=command)
            save(path/'manifest.json', manifest)
            print(now(), 'START', job['id'], 'log:', out/'stdout.log', flush=True)
            with (out/'stdout.log').open('w') as stream:
                child = subprocess.Popen(command, cwd=job['cwd'], env=environment(ROOT,job,gpu),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = child.pid
                save(path/'manifest.json', manifest)
                def terminate(signum, frame):
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    job.update(state='INTERRUPTED', ended_at=now())
                    manifest['state'] = 'INTERRUPTED'
                    save(path/'manifest.json', manifest)
                    raise SystemExit(128+signum)
                previous_handlers = {}
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    previous_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, terminate)
                exit_code = child.wait()
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
            report = json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
            valid = exit_code == 0 and report.get('state') in ('COMPLETED','VERIFIED')
            job.update({k:report.get(k) for k in ['episodes','successes','success_rate','policy_ms_mean','model_ms_mean','measured_calls',
                'successful_calls','flops_eq9_prefill_T','flops_gated_prefill_T','flops_profiled_call_T','flops_profiled_warm_T',
                'flops_profiled_samples','flops_formula_calls','latency_profiled_calls_excluded']})
            job.update(state=report['state'] if valid else 'FAILED', exit_code=exit_code, ended_at=now())
            save(path/'manifest.json', manifest)
            write_summary(path, manifest['jobs'])
            print(now(), 'END', job['id'], job['state'], flush=True)
            if not valid:
                manifest['state'] = 'FAILED'
                save(path/'manifest.json', manifest)
                raise ValueError('Condition failed; queue stopped. Inspect its stdout.log and result.json.')
        manifest.update(state='COMPLETE', ended_at=now())
        save(path/'manifest.json', manifest)
        (path/'COMPLETE').touch()
        print('COMPLETE:', path, flush=True)

def main():
    args = parser().parse_args()
    if args.list:
        print(json.dumps({'models':MODELS, 'strategies':STRATEGIES, 'suites':SUITES,
                          'backends':json.loads((ROOT/'configs/backends.json').read_text())}, indent=2))
        return
    if args.status:
        status(args.status)
        return
    if args.pause:
        path = run_path(args.pause)
        assert (path/'manifest.json').is_file()
        (path/'PAUSE_REQUEST').write_text(now()+'\n')
        print('Pause requested:', path)
        return
    if args.list_tasks or args.doctor:
        command = [sys.executable, str(ROOT/'tools/inspect_environment.py')]
        command += ['--doctor'] if args.doctor else ['--suite', *suite_keys(args.suite)]
        raise SystemExit(subprocess.call(command))
    if args.verify_calls < 1 or args.warmup_calls < 0 or args.gpu < 0 or args.flops_sample_interval < 0:
        raise ValueError('Invalid call count or GPU index.')
    if args.resume:
        path = run_path(args.resume)
        manifest = json.loads((path/'manifest.json').read_text())
        if args.tmux or args.dry_run:
            raise ValueError('--resume does not accept --tmux or --dry-run; use an existing tmux shell if desired.')
    else:
        jobs = make_jobs(ROOT, args)
        for job in jobs:
            validate_job(ROOT, job)
        if args.dry_run:
            print(json.dumps({'jobs':jobs, 'conditions':len(jobs),
                              'expected_episodes':sum(j['expected_episodes'] for j in jobs) if args.mode=='eval' else 0}, indent=2))
            return
        name = args.name or datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name):
            raise ValueError('--name must be a safe short directory name, not a path.')
        path = ROOT/'runs'/name
        path.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version':1, 'created_at':now(), 'state':'PLANNED', 'gpu':args.gpu,
                    'save_video':args.save_video, 'jobs':jobs,
                    'method_notes':'Source adapters preserved; cache ratios mean reuse, not dropped tokens. See README.'}
        snapshot=path/'code_snapshot'
        shutil.copytree(ROOT/'pact_eval',snapshot/'pact_eval',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'evaluate.py',snapshot/'evaluate.py')
        shutil.copytree(ROOT/'configs',snapshot/'configs')
        if (ROOT/'docs').is_dir():
            shutil.copytree(ROOT/'docs',snapshot/'docs')
        shutil.copy2(ROOT/'README.md',snapshot/'README.md')
        manifest['worker_snapshot']=str(snapshot/'pact_eval/worker.py')
        manifest['creation_argv']=sys.argv
        save(path/'manifest.json', manifest)
        write_summary(path, jobs)
        if args.tmux:
            session = 'pact_'+re.sub(r'[^A-Za-z0-9_-]', '_',name)
            argv = [sys.executable, str(ROOT/'evaluate.py'), '--resume', str(path)]
            launch = shlex.join(argv)+' > '+shlex.quote(str(path/'queue.log'))+' 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',session,launch], check=True)
            print('Started tmux session:',session)
            print('Run:',path)
            print('View:',shlex.join(['tmux','attach','-t',session]))
            print('Status:',shlex.join([sys.executable,str(ROOT/'evaluate.py'),'--status',str(path)]))
            return
    execute(path, manifest, args.retry_failed)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, BlockingIOError) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(2)
