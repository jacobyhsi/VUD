"""Where each model's VUD output goes wrong: total vs aleatoric vs epistemic, against the true p(y|x).
Total: KL(true || model) on p(y|x) and overconfidence (TU below the true aleatoric). Aleatoric: |Va - true|
on and off the true boundary (true aleatoric > 0.3 bits). Epistemic: Ve near/away and its target TU - true."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1
MODELS = ['MDLM-TabPFN-ControlledGP50-0.8M', 'MDLM-TabPFNClassify2D-Geometry-0.8M', 'MDLM-TabPFNClassify2D-0.8M',
          'MDLM-TabPFNFullClassify2D-0.8M', 'MDLM-BayesClassify2D-2.5M', 'Dream-v0-Base-7B']
bits = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
for ds in ('moons_1', 'logistic_regression'):
    cols = ['x1'] if ds == 'logistic_regression' else ['x1', 'x2']
    print(f'== {ds}')
    print(f'{"model":36s} | TOTAL: KL(true||model) near/away, overconf% | ALEATORIC |Va-true|: on-bnd/off-bnd near, away | EPISTEMIC Ve near->away (target near->away)')
    for m in MODELS:
        df = pd.read_csv(f'results/toy_classification/{ds}/{m}/va_vanilla/results_test.csv')
        cx = pd.read_csv(f'results/toy_classification/{ds}/{m}/va_vanilla/D_test.csv')[cols].to_numpy(float)
        rows = []
        for key, g in df.groupby([f'x_{c}' for c in cols]):
            cand = g.loc[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))]
            s = cand.loc[cand.Va.idxmin()]
            h = g['H[p(y|x,D)]'].iloc[0]
            va = s.Va if len(cols) == 2 else min(s.Va, h)
            rows.append([*np.atleast_1d(key), g['p(y=1|x,D)'].iloc[0], h, va])
        a = np.array(rows); x = a[:, :len(cols)]; p, tu, va = a[:, len(cols):].T
        pt = np.clip(true_p1(ds, x), 1e-9, 1 - 1e-9); p = np.clip(p, 1e-9, 1 - 1e-9)
        ale = bits(pt); ve = tu - va
        kl = pt * np.log2(pt / p) + (1 - pt) * np.log2((1 - pt) / (1 - p))
        near = ((x[:, 0] >= cx[:, 0].min()) & (x[:, 0] <= cx[:, 0].max())) if len(cols) == 1 else np.linalg.norm(x[:, None] - cx[None], axis=-1).min(1) < .35
        bnd = ale > 0.3
        over = (tu < ale - 0.1).mean() * 100
        err = np.abs(va - ale)
        on = f'{err[near & bnd].mean():.2f}' if (near & bnd).any() else '  - '
        print(f'{m.replace("MDLM-", "")[:36]:36s} | {kl[near].mean():.2f}/{kl[~near].mean():.2f}, {over:4.1f}% | '
              f'{on}/{err[near & ~bnd].mean():.2f}, {err[~near].mean():.2f} | '
              f'{ve[near].mean():.2f}->{ve[~near].mean():.2f} ({(tu - ale)[near].mean():.2f}->{(tu - ale)[~near].mean():.2f})')
