#!/usr/bin/env python3
"""Restore verified pre-test metadata, preserving both versions. Never weights."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PROV=ROOT/'provenance/migration_20260831'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before=json.loads((PROV/'vendor__vla-cache.before.json').read_text())
    base=ROOT/'vendor/vla-cache'
    changed=[]
    for name,meta in before.items():
        if '/checkpoints/' not in name or 'sha256' not in meta:
            continue
        path=base/name
        if path.suffix not in ('.py','.json') or not path.is_file() or sha(path)==meta['sha256']:
            continue
        assert path.resolve().is_relative_to(base.resolve()) and not path.is_symlink()
        backups=[p for p in path.parent.glob(path.name+'.back.*') if sha(p)==meta['sha256']]
        if not backups:
            raise RuntimeError(f'No verified backup; no restoration performed: {path}')
        changed.append((path,sorted(backups)[-1],meta['sha256']))
    records=[]
    for path,backup,expected in changed:
        saved=ROOT/'provenance/loader_side_effect_before_isolation'/path.relative_to(base)
        saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():
            raise RuntimeError(f'Refusing to overwrite saved version: {saved}')
        shutil.copy2(path,saved)
        shutil.copy2(backup,path)
        assert sha(path)==expected
        records.append(dict(restored=str(path),backup=str(backup),source_sha256=expected,
            loader_generated_version_preserved=str(saved),restored_exact=True))
    out=ROOT/'provenance/loader_side_effect_restoration.json'
    if out.exists():
        records=json.loads(out.read_text())+records
    out.write_text(json.dumps(records,indent=2))
    print('RESTORED_CHECKPOINT_METADATA',json.dumps(records))


if __name__=='__main__':
    main()
