"""Ideal VUD (exact posterior) under the controlled-GP50 prior with 50% vs 30% clean tasks, against the truth."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from evaluate_vud_classification import true_p1
bits = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
for ds in ('logistic_regression', 'moons_1'):
    cols = ['x1'] if ds == 'logistic_regression' else ['x1', 'x2']
    cx = pd.read_csv(f'../results/toy_classification/{ds}/Dream-v0-Base-7B/va_vanilla/D_test.csv')[cols].to_numpy(float)
    print(f'== {ds}  (ideal VUD = VUD run on the exact posterior; Va vs true aleatoric, bits)')
    for tag, folder in (('50% clean (current)', ''), ('30% clean', '_clean0.3')):
        ref = pd.read_csv(f'tmp/logs/conditional_reference_{ds}{folder}/reference.csv')
        g = ref.groupby(cols).agg(TU=('bayes_TU', 'first'), Va=('bayes_Va', 'min')).reset_index()
        x = g[cols].to_numpy()
        ale = bits(np.clip(true_p1(ds, x), 1e-9, 1 - 1e-9))
        near = ((x[:, 0] >= cx[:, 0].min()) & (x[:, 0] <= cx[:, 0].max())) if len(cols) == 1 else np.linalg.norm(x[:, None] - cx[None], axis=-1).min(1) < .35
        va, ve = g.Va.to_numpy(), (g.TU - g.Va).to_numpy()
        extra = ''
        if len(cols) == 2:
            bnd = ale > 0.3
            extra = f' | near off-boundary Va {va[near & ~bnd].mean():.2f} (true {ale[near & ~bnd].mean():.2f})'
        print(f'   {tag:20s} near: Va {va[near].mean():.2f} (true {ale[near].mean():.2f}, err {np.abs(va - ale)[near].mean():.2f}), Ve {ve[near].mean():.2f} | '
              f'away: Va {va[~near].mean():.2f} (true {ale[~near].mean():.2f}, err {np.abs(va - ale)[~near].mean():.2f}), Ve {ve[~near].mean():.2f}{extra}')
