"""Summarise the simulated ground-truth comparison (classification): per model and near/far queries (bits)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

d = pd.concat([pd.read_csv(f) for f in sorted(Path(sys.argv[1]).glob('queries_*.csv'))])
H = lambda p: -(np.clip(p, 1e-12, 1) * np.log2(np.clip(p, 1e-12, 1)) + np.clip(1 - p, 1e-12, 1) * np.log2(np.clip(1 - p, 1e-12, 1)))
xent = lambda pt, pm: -(pt * np.log2(np.clip(pm, 1e-12, 1)) + (1 - pt) * np.log2(np.clip(1 - pm, 1e-12, 1)))
models = sorted({c[:-3] for c in d.columns if c.endswith('_p1') and c != 'bayes_p1'})
print(f'{d.task.nunique()} tasks, {len(d)} queries; MLP effective sample size median {d.ess_mlp.median():.0f}')
for q in ('near', 'far'):
    s = d[d['query'] == q]
    print(f'\n== {q} queries (n={len(s)}): truth -> true aleatoric {s.true_AU.mean():.3f}')
    print(f'   {"":22s} | KL(true||pred) | |p - Bayes p| | TU    | AU est (|err| vs true) | EU est (|err| vs Bayes EU)')
    print(f'   {"exact Bayes":22s} | {np.mean(xent(s.p_true, s.bayes_p1) - H(s.p_true)):14.3f} | {0:13.3f} | {s.bayes_TU.mean():.3f} | '
          f'{s.bayes_AU.mean():.3f} ({np.abs(s.bayes_AU - s.true_AU).mean():.3f})          | {s.bayes_EU.mean():.3f} (0)')
    print(f'   {"VUD on exact Bayes":22s} | {"":14s} | {"":13s} | {s.bayes_TU.mean():.3f} | '
          f'{s.ideal_Va.mean():.3f} ({np.abs(s.ideal_Va - s.true_AU).mean():.3f})          | {s.ideal_Ve.mean():.3f} ({np.abs(s.ideal_Ve - s.bayes_EU).mean():.3f})')
    for m in models:
        print(f'   {"MDLM " + m + " + VUD":22s} | {np.mean(xent(s.p_true, s[m + "_p1"]) - H(s.p_true)):14.3f} | {np.abs(s[m + "_p1"] - s.bayes_p1).mean():13.3f} | '
              f'{s[m + "_TU"].mean():.3f} | {s[m + "_Va"].mean():.3f} ({np.abs(s[m + "_Va"] - s.true_AU).mean():.3f})          | '
              f'{s[m + "_Ve"].mean():.3f} ({np.abs(s[m + "_Ve"] - s.bayes_EU).mean():.3f})')
    for m in models:
        r_au = np.corrcoef(s[m + '_Va'], s.true_AU)[0, 1]; r_eu = np.corrcoef(s[m + '_Ve'], s.bayes_EU)[0, 1]
        print(f'   correlation across queries, {m}: Va vs true AU {r_au:.2f} | Ve vs Bayes EU {r_eu:.2f}')
