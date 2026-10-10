"""Once both MDLM-BayesRegress-0.8M VUD runs have all 150 x, append their PNGs to the completed-PNG zip."""
import collections, csv, subprocess, time
from pathlib import Path
from zipfile import ZipFile

root = Path('/home/js2723/PROJECTS/VUD')
archive = root / 'results_pngs_completed_20260929T072416Z.zip'
folders = [root / 'results/toy_regression/gaps/MDLM-BayesRegress-0.8M/va_vanilla/seed_0',
           root / 'results/toy_regression/linear_regression/MDLM-BayesRegress-0.8M/va_vanilla/seed_0']


def complete(folder):
    counts = collections.Counter(r['x_note'] for r in csv.DictReader(open(folder / 'results_test.csv')))
    return len(counts) == 150 and set(counts.values()) == {5}


def running():
    out = subprocess.run(['timeout', '30', 'srun', '--jobid=76856', '--overlap', '--cpu-bind=none',
                          'ps', '-o', 'args', '-u', 'js2723'], capture_output=True, text=True).stdout
    return 'MDLM-BayesRegress-0.8M/va_vanilla' in out


while not all(complete(f) for f in folders) or running():
    time.sleep(60)
files = sorted(p for f in folders for p in f.glob('*.png'))
with ZipFile(archive, 'a') as bundle:
    existing = set(bundle.namelist())
    for path in files:
        name = str(path.relative_to(root))
        if name not in existing:
            bundle.write(path, name)
with ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    print(f'{archive}: {len(bundle.namelist())} PNGs')
    print('\n'.join(n for n in bundle.namelist() if 'MDLM-BayesRegress' in n))
