from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from csaps import csaps

root = Path('results/toy_classification/logistic_regression/Dream-v0-Base-7B')
runs = {}
for folder in sorted(root.glob('va_*')):
    path = folder / 'results_test.csv'
    if not path.exists():
        continue
    df = pd.read_csv(path)
    rows = []
    for x, g in df.groupby('x_x1'):
        kl = g['kl_pyx_pyxz']
        h = g['H[p(y|x,D)]'].iloc[0]
        va = min(g.loc[kl.isin(kl.nsmallest(5)), 'Va'].min(), h)
        rows.append((round(x, 4), h, va, h-va, len(g)))
    frame = pd.DataFrame(rows, columns=['x','H','Va','Ve','nz']).set_index('x')
    runs[folder.name] = frame
    n = sorted(df.n_joint_samples.unique()) if 'n_joint_samples' in df else 'vanilla'
    digest = hashlib.sha256((folder / 'D_test.csv').read_bytes()).hexdigest()[:10]
    print('RUN', folder.name, 'N', n, 'points', len(frame), 'range', (frame.index.min(),frame.index.max()), 'zcounts',sorted(frame.nz.unique()), 'D',digest, flush=True)

base = runs['va_vanilla']
common = sorted(set.intersection(*(set(d.index) for d in runs.values())))
print('COMMON', len(common), min(common), max(common), flush=True)
for name,d in runs.items():
    if name == 'va_vanilla':
        continue
    print('METRICS', name, flush=True)
    for tag,xs in [('available', sorted(set(d.index)&set(base.index))),('common',common)]:
        a,b = d.loc[xs],base.loc[xs]
        err = (a[['H','Va','Ve']]-b[['H','Va','Ve']]).abs().mean()
        bias = (a.Va-b.Va).mean()
        grid = np.linspace(min(xs),max(xs),300)
        smooth_mae = np.mean(np.abs(csaps(xs,a.Va,grid,smooth=.85)-csaps(xs,b.Va,grid,smooth=.85)))
        print(tag, 'MAE',err.round(5).to_dict(),'Va_bias',round(bias,5),'smooth_Va_MAE',round(smooth_mae,5),flush=True)
