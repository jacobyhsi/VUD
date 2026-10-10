from pathlib import Path
from datetime import datetime, timezone
import subprocess

root = Path('/home/js2723/PROJECTS/VUD/results/toy_classification/moons_1/Dream-v0-Base-7B')
for method in ('va_vanilla', 'va_parallel_pp'):
    files = sorted((root/method).glob('results_test_x*.csv'), key=lambda p: p.stat().st_mtime)
    print('moons_1', method, len(files), '/400', flush=True)
    if files:
        print('latest', datetime.fromtimestamp(files[-1].stat().st_mtime, timezone.utc).isoformat(), flush=True)
    if len(files) > 10:
        seconds = (files[-1].stat().st_mtime-files[-11].stat().st_mtime)/10
        print('recent seconds/point', round(seconds), 'remaining hours', round((400-len(files))*seconds/3600, 1), flush=True)
subprocess.run(['nvidia-smi'])
subprocess.run(['ps', '-ww', '-u', 'js2723', '-o', 'pid,etime,args'])
