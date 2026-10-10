"""Summarise tabpfn_original_eval.py outputs: per model, overall and by distance of the test point from the context."""
import sys
import numpy as np, pandas as pd

d = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])
import os
models = os.environ.get('SUMMARY_MODELS', '').split(',') if os.environ.get('SUMMARY_MODELS') else \
    [c for c in d.columns if c not in ('dataset', 'd', 'n', 'dist', 'y')]
H = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


print(f'{d.dataset.nunique()} datasets, {len(d)} test points; 1-feature datasets {(d.groupby("dataset").d.first() == 1).mean():.0%}')
for name, s in [('all test points', d), ('inside the context (< 1 std)', d[d.dist < 1]),
                ('edge (1-2 std)', d[(d.dist >= 1) & (d.dist < 2)]), ('outside (>= 2 std)', d[d.dist >= 2])]:
    print(f'\n== {name}: {len(s)} points, class-1 share {s.y.mean():.2f}')
    print(f'   {"model":20s} | log-loss (bits) | accuracy | ECE   | TU (bits) | |p - p_TabPFN|')
    for m in models:
        p = s[m].clip(1e-6, 1 - 1e-6).to_numpy(); y = s.y.to_numpy()
        ll = -(y * np.log2(p) + (1 - y) * np.log2(1 - p)).mean()
        print(f'   {m:20s} | {ll:15.3f} | {((p > .5) == y).mean():8.3f} | {ece(p, y):.3f} | {H(p).mean():9.3f} | {np.abs(p - s.TabPFN_v1).mean():.3f}')
