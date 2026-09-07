#!/usr/bin/env python3
"""One-time, allowlisted, inode-preserving server migration with old-path aliases.

No experiment or model file is deleted. Linux renameat2(RENAME_EXCHANGE) swaps
each original path with its prepared compatibility symlink atomically, so old
editable installs and open working directories remain valid throughout.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
from datetime import datetime

ROOT = Path('/home/ubuntu/PACT-VLA')
TARGETS = [
    ('/home/ubuntu/ccw/VLA-Pruner', 'vendor/vla-pruner'),
    ('/home/ubuntu/ccw/vla-cache', 'vendor/vla-cache'),
    ('/home/ubuntu/ccw/divprune-official', 'vendor/divprune-official'),
    ('/home/ubuntu/LIBERO', 'dependencies/LIBERO'),
    ('/home/ubuntu/openvla-oft-official-e4287e9', 'references/openvla-oft-official-e4287e9'),
    ('/home/ubuntu/transformers-openvla-oft-official', 'references/transformers-openvla-oft-official'),
    ('/home/ubuntu/openvla-oft-official-e4287e9.tar.gz', 'references/downloads/openvla-oft-official-e4287e9.tar.gz'),
    ('/home/ubuntu/transformers-openvla-oft-official.tar.gz', 'references/downloads/transformers-openvla-oft-official.tar.gz'),
    ('/home/ubuntu/ccw/vla_pruner.pdf', 'references/vla_pruner.pdf'),
    ('/home/ubuntu/vla-pruner-clean-84d4', 'references/vla-pruner-clean-84d4'),
    ('/home/ubuntu/vlapruner_robotwin_eval_20260827_1407', 'experiments/legacy/robotwin_20260827'),
    ('/home/ubuntu/SimplerEnv-OpenVLA', 'experiments/legacy/SimplerEnv-OpenVLA'),
]

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def nodes(path):
    if not path.is_dir():
        return [path]
    result = []
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in sorted(dirs + files):
            result.append(Path(base)/name)
    return sorted(result)

def inventory(path):
    records = {}
    for p in nodes(path):
        s = p.lstat()
        key = str(p.relative_to(path)) if path.is_dir() else '.'
        entry = dict(size=s.st_size, mode=s.st_mode, mtime_ns=s.st_mtime_ns,
                     inode=s.st_ino, device=s.st_dev)
        if p.is_symlink():
            entry['link'] = os.readlink(p)
        elif stat.S_ISREG(s.st_mode) and s.st_size <= 2*1024*1024:
            entry['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        records[key] = entry
    return records

def occupied():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            text=True, capture_output=True, check=True)
    return result.stdout.strip()

def exchange(a, b):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(a), -100, os.fsencode(b), 2):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(a), str(b))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert ROOT.is_absolute() and ROOT == Path('/home/ubuntu/PACT-VLA')
    planned = [{'old': a, 'new': str(ROOT/b)} for a,b in TARGETS]
    if not args.apply:
        print(json.dumps(planned, indent=2))
        return
    ROOT.mkdir(exist_ok=True)
    assert not ROOT.is_symlink() and ROOT.resolve() == ROOT
    prov = ROOT/'provenance/migration_20260831'
    prov.mkdir(parents=True, exist_ok=True)
    with (prov/'migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if occupied():
            raise SystemExit('GPU is occupied; no migration performed. Retry when the run is finished.')
        journal_path = prov/'journal.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {'started_at': now(), 'moves': []}
        backup = prov/'source_and_log_backup.tar.gz'
        if not backup.exists():
            with tarfile.open(backup, 'w:gz') as archive:
                for old, rel in TARGETS[:3]:
                    source = Path(old)
                    if source.is_symlink():
                        continue
                    for p in nodes(source):
                        if not p.is_file() or p.is_symlink() or p.stat().st_size > 2*1024*1024:
                            continue
                        if p.suffix.lower() not in {'.py','.sh','.json','.md','.txt','.yaml','.yml','.toml','.csv','.tsv','.patch','.ipynb'}:
                            continue
                        if any(part in {'checkpoints','__pycache__','.git'} for part in p.relative_to(source).parts):
                            continue
                        archive.add(p, arcname=str(Path(rel)/p.relative_to(source)), recursive=False)
            print('SOURCE_BACKUP_COMPLETE', backup, flush=True)
        for old, rel in TARGETS:
            source, target = Path(old), ROOT/rel
            assert target.is_relative_to(ROOT) and target != ROOT
            if source.is_symlink() and source.resolve() == target and target.exists():
                prior = next((m for m in journal['moves'] if m['old'] == old), None)
                if prior and not prior['exact_inventory_match']:
                    tag = rel.replace('/', '__')
                    a = json.loads((prov/(tag+'.before.json')).read_text())
                    b = json.loads((prov/(tag+'.after.json')).read_text())
                    assert a.keys() == b.keys()
                    differences = [k for k in a if a[k] != b[k]]
                    # git status in the first version created/removed index.lock,
                    # updating only the .git DIRECTORY mtime, not any file.
                    assert differences == ['.git']
                    assert {k:v for k,v in a['.git'].items() if k != 'mtime_ns'} == {
                        k:v for k,v in b['.git'].items() if k != 'mtime_ns'}
                    prior['verification_resolution'] = 'Only .git directory mtime changed from git status lock; all files, hashes, inodes and links match exactly.'
                    prior['content_inode_verified'] = True
                    save(journal_path, journal)
                print('ALREADY_MOVED', old, flush=True)
                continue
            if not source.exists() or source.is_symlink():
                raise SystemExit(f'Unexpected/missing source: {source}')
            if source.resolve() != source:
                raise SystemExit(f'Source ancestor unexpectedly redirected: {source}')
            if target.exists() or target.is_symlink():
                raise SystemExit(f'Refusing to overwrite {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            assert source.stat().st_dev == target.parent.stat().st_dev, 'Migration must stay on one filesystem'
            if occupied():
                raise SystemExit('A new GPU job started; remaining migration deferred.')
            tag = rel.replace('/', '__')
            if (source/'.git').is_dir():
                git = {}
                for name, argv in [('head', ['rev-parse','HEAD']), ('status', ['status','--porcelain=v1','--untracked-files=all'])]:
                    result = subprocess.run(['git','-C',str(source),*argv], capture_output=True, text=True,
                                            env={**os.environ, 'GIT_OPTIONAL_LOCKS':'0'})
                    git[name] = result.stdout
                    git[name+'_exit'] = result.returncode
                save(prov/(tag+'.git.json'), git)
            before = inventory(source)
            save(prov/(tag+'.before.json'), before)
            # The temporary self-link lives only at the unused new destination.
            # One kernel operation publishes the new directory and old-path alias.
            os.symlink(str(target), target, target_is_directory=source.is_dir())
            try:
                exchange(source, target)
            except BaseException:
                if target.is_symlink() and os.readlink(target) == str(target):
                    target.unlink()  # Only our fresh self-link, never source data.
                raise
            assert source.is_symlink() and source.resolve() == target
            after = inventory(target)
            save(prov/(tag+'.after.json'), after)
            exact = before == after
            entry = dict(old=old, new=str(target), at=now(), entries=len(after),
                         exact_inventory_match=exact,
                         content_inode_verified=exact,
                         regular_file_bytes=sum(v['size'] for v in after.values() if stat.S_ISREG(v['mode'])),
                         old_path_alias=True)
            journal['moves'].append(entry)
            save(journal_path, journal)
            print('MOVED', json.dumps(entry), flush=True)
            if not exact:
                raise SystemExit('Inventory changed during migration; both paths remain valid. Inspect before proceeding.')
        alias = ROOT/'LIBERO'
        if not alias.exists():
            alias.symlink_to('dependencies/LIBERO', target_is_directory=True)
        for family, subdir in [('openvla','openvla'),('oft','openvla-oft')]:
            link = ROOT/'checkpoints'/family
            link.parent.mkdir(exist_ok=True)
            if not link.exists():
                link.symlink_to('../vendor/vla-pruner/src/'+subdir+'/checkpoints', target_is_directory=True)
        # Rebase shared checkpoint symlinks to their new canonical targets.
        # Preserve the original link audit across idempotent or extended runs.
        link_journal = prov/'rebased_links.json'
        rebased = json.loads(link_journal.read_text()) if link_journal.exists() else []
        for base in (ROOT/'vendor/vla-cache/src',):
            for p in nodes(base):
                if not p.is_symlink():
                    continue
                old_link = os.readlink(p)
                if old_link.startswith('/home/ubuntu/ccw/VLA-Pruner/'):
                    new_target = ROOT/'vendor/vla-pruner'/old_link.split('/home/ubuntu/ccw/VLA-Pruner/',1)[1]
                    assert new_target.exists() and new_target.is_relative_to(ROOT)
                    new_link = os.path.relpath(new_target, p.parent)
                    temp = p.with_name(p.name+'.pact-migration-link')
                    assert not temp.exists() and not temp.is_symlink()
                    temp.symlink_to(new_link)
                    temp.replace(p)
                    rebased.append(dict(path=str(p), before=old_link, after=new_link))
        save(prov/'rebased_links.json', rebased)
        journal['completed_at'] = now()
        journal['state'] = 'COMPLETE'
        save(journal_path, journal)
        print('MIGRATION_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
