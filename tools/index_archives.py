#!/usr/bin/env python3
"""Create discoverable links and inventories; never execute historical queues."""
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pact_eval.planning import backend, environment


def main():
    records = []
    for repo in ('vla-pruner','vla-cache','divprune-official'):
        source = ROOT/'vendor'/repo
        archive = ROOT/'experiments/archive'/repo
        archive.mkdir(parents=True,exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.name.startswith('.') or item.name in ('src','__pycache__'):
                continue
            link = archive/item.name
            target = os.path.relpath(item,link.parent)
            if link.is_symlink():
                if link.resolve()!=item.resolve():
                    raise ValueError(f'Unexpected existing link: {link}')
            elif link.exists():
                raise ValueError(f'Refusing to overwrite {link}')
            else:
                link.symlink_to(target,target_is_directory=item.is_dir())
            records.append(dict(repository=repo,name=item.name,path=str(item),archive_link=str(link)))
    provenance = ROOT/'provenance'
    provenance.mkdir(exist_ok=True)
    (provenance/'archive_index.json').write_text(json.dumps(records,indent=2))
    envdir = provenance/'environments_20260831'
    envdir.mkdir(exist_ok=True)
    inventory=[]
    for kind in ('native','vla-cache'):
        for family in ('openvla','oft'):
            spec = backend(ROOT,family,kind)
            name = spec['environment']
            result = subprocess.run([spec['python'],'-m','pip','freeze'],
                env=environment(ROOT,spec,0),cwd=spec['cwd'],text=True,capture_output=True,check=True)
            (envdir/(name+'.pip-freeze.txt')).write_text(result.stdout)
            conda = Path(spec['python']).parents[3]/'bin/conda'
            result = subprocess.run([str(conda),'env','export','-n',name,'--no-builds'],text=True,capture_output=True,check=True)
            (envdir/(name+'.environment.yml')).write_text(result.stdout)
            inventory.append(dict(family=family,kind=kind,**spec))
    (envdir/'routing.json').write_text(json.dumps(inventory,indent=2))
    print(f'Indexed {len(records)} preserved items; exported four environments to {envdir}')


if __name__=='__main__':
    main()
