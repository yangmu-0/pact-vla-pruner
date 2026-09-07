#!/usr/bin/env python3
"""Read-only checks: no model load, downloads or package installation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pact_eval.planning import backend, environment, SUITES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doctor', action='store_true')
    parser.add_argument('--suite', nargs='+', default=['spatial'])
    args = parser.parse_args()
    if args.doctor:
        results = []
        for kind in ('native','vla-cache'):
            for family in ('openvla','oft'):
                spec = backend(ROOT, family, kind)
                code = "import json,sys,torch,transformers,libero; print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,transformers_path=transformers.__file__,libero_path=libero.__file__)))"
                check = subprocess.run([spec['python'],'-c',code], env=environment(ROOT,spec,0),
                    cwd=spec['cwd'], text=True, capture_output=True)
                cps = []
                for _,suffix in SUITES.values():
                    cp = Path(spec['cwd'])/'checkpoints'/('openvla-7b-'+('oft-' if family=='oft' else '')+'finetuned-libero-'+suffix)
                    required = ['config.json','dataset_statistics.json']
                    index = cp/'model.safetensors.index.json'
                    required += list(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
                    missing = [name for name in required if not (cp/name).is_file()]
                    if family=='oft':
                        missing += [prefix+'*.pt' for prefix in ('action_head','proprio_projector')
                            if not any(p.is_file() for p in cp.glob(prefix+'*.pt'))]
                    cps.append(dict(path=str(cp), missing=missing))
                results.append(dict(backend=kind+'_'+family, returncode=check.returncode,
                    output=check.stdout.strip(), stderr=check.stderr.strip(), checkpoints=cps))
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return int(any(r['returncode'] or any(c['missing'] for c in r['checkpoints']) for r in results))
    spec = backend(ROOT,'openvla','native')
    suites = [SUITES[key][0] for key in args.suite]
    code = ('import json,contextlib,sys; output={}\n'
            'with contextlib.redirect_stdout(sys.stderr):\n'
            ' from libero.libero import benchmark\n'
            ' factories=benchmark.get_benchmark_dict()\n'
            f' for name in {suites!r}:\n'
            '  s=factories[name](); output[name]=[dict(id=i,name=s.get_task(i).name,language=s.get_task(i).language) for i in range(s.n_tasks)]\n'
            'print(json.dumps(output,indent=2))')
    return subprocess.call([spec['python'],'-c',code], env=environment(ROOT,spec,0), cwd=spec['cwd'])


if __name__ == '__main__':
    raise SystemExit(main())
