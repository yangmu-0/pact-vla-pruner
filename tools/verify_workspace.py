#!/usr/bin/env python3
"""Audit delivered artifacts without starting experiments or changing sources."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pact_eval.worker import episode_counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Write provenance/delivery_verification.json')
    args=parser.parse_args()
    errors=[]
    journal=json.loads((ROOT/'provenance/migration_20260831/journal.json').read_text())
    for record in journal['moves']:
        old,new=Path(record['old']),Path(record['new'])
        if not old.is_symlink() or old.resolve()!=new or not new.exists() or not record.get('content_inode_verified'):
            errors.append('Migration/alias not verified: '+str(old))
    rebased=json.loads((ROOT/'provenance/migration_20260831/rebased_links.json').read_text())
    for entry in rebased:
        if not Path(entry['path']).is_file():
            errors.append('Broken shared checkpoint: '+entry['path'])
    original_code_checked=0
    original_weights_checked=0
    for repository in ('vla-pruner','vla-cache'):
        base=ROOT/'vendor'/repository
        snapshot=json.loads((ROOT/'provenance/migration_20260831'/('vendor__'+repository+'.before.json')).read_text())
        for name,meta in snapshot.items():
            file=base/name
            if not name.startswith('src/') or not file.is_file() or file.is_symlink():
                continue
            if 'sha256' in meta and file.suffix in ('.py','.sh','.json'):
                original_code_checked+=1
                if hashlib.sha256(file.read_bytes()).hexdigest()!=meta['sha256']:
                    errors.append('Original source/metadata changed: '+str(file))
            if '/checkpoints/' in name and file.suffix in ('.safetensors','.pt','.bin'):
                original_weights_checked+=1
                current=file.stat()
                if (current.st_ino,current.st_size,current.st_mtime_ns)!=(meta['inode'],meta['size'],meta['mtime_ns']):
                    errors.append('Original weight identity/size/mtime changed: '+str(file))
    records=[]
    for manifest_path in sorted((ROOT/'runs').glob('integration_*/manifest.json')):
        manifest=json.loads(manifest_path.read_text())
        if manifest['state']!='COMPLETE':
            errors.append('Integration run incomplete: '+str(manifest_path))
        for job in manifest['jobs']:
            out=Path(job['output']) if job.get('output') else None
            if out is None or not (out/'result.json').is_file():
                errors.append('Missing result: '+job['id'])
                continue
            result=json.loads((out/'result.json').read_text())
            tasks=json.loads((out/'tasks.json').read_text())
            if [t['id'] for t in tasks]!=job['task_ids']:
                errors.append('Task selection mismatch: '+job['id'])
            if result['policy_errors'] or result['episode_errors']:
                errors.append('Policy/episode errors: '+job['id'])
            rows=list(csv.DictReader((out/'policy_timing_calls.csv').open()))
            if len(rows)!=result['successful_calls']:
                errors.append('Timing denominator mismatch: '+job['id'])
            episodes,successes,_=episode_counts(out)
            if (episodes,successes)!=(result['episodes'],result['successes']):
                errors.append('Episode parser mismatch: '+job['id'])
            if job['mode']=='eval' and (result['state']!='COMPLETED' or episodes!=job['expected_episodes']):
                errors.append('Eval incomplete: '+job['id'])
            if job['mode']=='verify' and (result['state']!='VERIFIED' or result['successful_calls']!=job['verify_calls']):
                errors.append('Verification incomplete: '+job['id'])
            n=512 if job['model']=='oft' else 256
            permitted={n} if job['strategy']=='vanilla' else {round(n*(1-job['ratio']))}
            if job['strategy'] in ('vla-pruner','vla-cache'):
                permitted.add(n)
            if not set(result['observed_visual_kept']).issubset(permitted):
                errors.append('Unexpected token budget: '+job['id'])
            records.append(dict(run=manifest_path.parent.name,id=job['id'],mode=job['mode'],
                state=result['state'],episodes=episodes,successful_calls=len(rows),
                task_ids=job['task_ids'],observed_visual_kept=result['observed_visual_kept']))
    stopped=ROOT/'vendor/vla-cache/table1_vlacache_4levels_trials20_20260830/PAUSED.json'
    old_pause=json.loads(stopped.read_text())
    if not old_pause.get('explicit_resume_required'):
        errors.append('Historical user-stop guard changed')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        text=True,capture_output=True,cwd=ROOT)
    if tests.returncode:
        errors.append('Unit tests failed')
    payload=dict(at=datetime.now().astimezone().isoformat(),state='PASS' if not errors else 'FAIL',
        migration_targets=len(journal['moves']),rebased_checkpoint_links=len(rebased),
        original_source_metadata_sha256_checks=original_code_checked,
        original_weight_inode_size_mtime_checks=original_weights_checked,
        integration_verify_conditions=sum(r['mode']=='verify' for r in records),
        integration_eval_conditions=sum(r['mode']=='eval' for r in records),
        completed_eval_episodes=sum(r['episodes'] for r in records if r['mode']=='eval'),
        unit_test_output=tests.stdout+tests.stderr,historical_cache_remains_paused=True,
        errors=errors,checks=records,
        limitations=['Integration checks are not success-rate reproduction.','No complete Table-1 experiment was started.'])
    if args.write:
        (ROOT/'provenance/delivery_verification.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))
    return int(bool(errors))


if __name__=='__main__':
    raise SystemExit(main())
