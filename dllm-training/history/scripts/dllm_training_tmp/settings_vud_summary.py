"""Professor's 3 settings on the classification toys: TU / AU (VUD Va) / EU (VUD Ve), near vs far from the data,
against the true aleatoric H(true p). Va = min over the 5 lowest-KL z, capped at TU (as in the toy experiments).
Near: within the context range (1-D) or within 0.35 of a context point (2-D)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1
MODELS = [('Setting 1: MDLM on notebook data', 'MDLM-TabPFNNotebook-0.8M'),
          ('Setting 2: MDLM on TabPFN-model data', 'MDLM-TabPFNModel-0.8M'),
          ('Setting 3: TabPFN v1 (as published)', 'TabPFN-v1'),
          ('ref: Dream-v0-Base-7B', 'Dream-v0-Base-7B')]
bits = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
for ds in ('logistic_regression', 'moons_1'):
    cols = ['x1'] if ds == 'logistic_regression' else ['x1', 'x2']
    print(f'\n== {ds}   (bits; AU error = mean |Va - true AU|)')
    print(f'   {"model":38s} | TU near / far | AU near / far (error)        | EU near / far')
    for label, m in MODELS:
        try:
            df = pd.read_csv(f'results/toy_classification/{ds}/{m}/va_vanilla/results_test.csv')
            cx = pd.read_csv(f'results/toy_classification/{ds}/{m}/va_vanilla/D_test.csv')[cols].to_numpy(float)
        except FileNotFoundError:
            print(f'   {label:38s} | (not run yet)'); continue
        rows = []
        for key, g in df.groupby([f'x_{c}' for c in cols]):
            cand = g.loc[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))]
            h = g['H[p(y|x,D)]'].iloc[0]
            rows.append([*np.atleast_1d(key), h, min(cand.Va.min(), h)])
        a = np.array(rows); x = a[:, :len(cols)]; tu, va = a[:, len(cols):].T
        ale = bits(np.clip(true_p1(ds, x), 1e-9, 1 - 1e-9)); ve = tu - va
        near = ((x[:, 0] >= cx[:, 0].min()) & (x[:, 0] <= cx[:, 0].max())) if len(cols) == 1 else \
            np.linalg.norm(x[:, None] - cx[None], axis=-1).min(1) < .35
        f = lambda v, s: v[s].mean()
        print(f'   {label:38s} | {f(tu, near):.2f} / {f(tu, ~near):.2f}   | {f(va, near):.2f} / {f(va, ~near):.2f} '
              f'({np.abs(va - ale)[near].mean():.2f} / {np.abs(va - ale)[~near].mean():.2f})   | {f(ve, near):.2f} / {f(ve, ~near):.2f}')
    print(f'   {"true AU":38s} |               | {f(ale, near):.2f} / {f(ale, ~near):.2f}')
