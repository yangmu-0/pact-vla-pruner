#!/usr/bin/env python3
"""Bounded integration checks: 12 single episodes, never a Table-1 rerun."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
tag=datetime.now().strftime('%Y%m%d_%H%M%S')
checks=[('goal','vla-pruner','50'),('object','vla-cache','75'),('long','divprune','87.5')]
for suite,strategy,ratio in checks:
    command=[sys.executable,str(ROOT/'evaluate.py'),'--model','all','--strategy',strategy,
        '--suite',suite,'--ratio',ratio,'--task-ids','3','--trials','1','--with-baseline',
        '--name',f'integration_episode_{tag}_{suite}']
    print('INTEGRATION_START',command,flush=True)
    result=subprocess.call(command)
    if result:
        print('INTEGRATION_FAILED',suite,result,flush=True)
        raise SystemExit(result)
print('INTEGRATION_COMPLETE',flush=True)
