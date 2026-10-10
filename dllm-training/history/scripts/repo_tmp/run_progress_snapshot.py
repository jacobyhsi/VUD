from pathlib import Path
from datetime import datetime, timezone
import csv
import subprocess

root = Path('/home/js2723/PROJECTS/VUD')
print(datetime.now(timezone.utc).isoformat(), flush=True)
for task, dataset, method in [
    ('toy_regression', 'gaps', 'va_vanilla/seed_0'),
    ('toy_regression', 'gaps', 'va_parallel_pp/seed_0'),
    ('toy_regression', 'linear_regression', 'va_vanilla/seed_0'),
    ('toy_regression', 'linear_regression', 'va_parallel_pp/seed_0'),
    ('toy_classification', 'spirals', 'va_vanilla'),
    ('toy_classification', 'spirals', 'va_parallel_pp'),
]:
    folder = root / 'results' / task / dataset / 'Dream-v0-Base-7B' / method
    files = sorted(folder.glob('results_test_x*.csv'), key=lambda p: p.stat().st_mtime)
    latest = datetime.fromtimestamp(files[-1].stat().st_mtime, timezone.utc).isoformat() if files else None
    print(dataset, method, 'count=', len(files), 'latest=', latest, flush=True)
    if len(files) > 11:
        seconds = (files[-1].stat().st_mtime - files[-11].stat().st_mtime) / 10
        target = 6400 if dataset == 'spirals' else 150
        print('recent_seconds_per_point=', round(seconds), 'remaining_hours=', round((target-len(files))*seconds/3600, 1), flush=True)
for path in sorted((root/'results/qa/Dream-v0-Instruct-7B/va_vanilla').glob('*.csv')):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    print('QA', path.name, 'rows=', len(rows), 'latest=', datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), flush=True)
subprocess.run(['ps', '-u', 'js2723', '-o', 'pid,etime,args'])
