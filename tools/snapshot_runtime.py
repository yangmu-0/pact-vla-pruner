#!/usr/bin/env python3
"""Archive custom installed code without moving or modifying Conda prefixes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pact_eval.planning import backend,environment


def hashes(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def main():
    records=[]
    for family in ('openvla','oft'):
        spec=backend(ROOT,family,'vla-cache')
        answer=subprocess.check_output([spec['python'],'-c',
            'import transformers; print(transformers.__path__[0])'],text=True,
            env=environment(ROOT,spec,0),cwd=spec['cwd']).strip()
        source=Path(answer)
        destination=ROOT/'references/runtime_sources'/spec['environment']/'transformers'
        if not destination.exists():
            shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        before,after=hashes(source),hashes(destination)
        if before!=after:
            raise RuntimeError(f'Runtime snapshot differs from installed package: {source}')
        records.append(dict(environment=spec['environment'],source=str(source),snapshot=str(destination),
            file_count=len(before),sha256=after,verified_identical=True,
            note='Archive only. Runtime routing still uses its validated installed environment.'))
    output=ROOT/'provenance/runtime_source_snapshots.json'
    output.write_text(json.dumps(records,indent=2))
    print('Verified runtime source snapshots:',[(r['environment'],r['file_count']) for r in records])


if __name__=='__main__':
    main()
