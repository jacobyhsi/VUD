"""moons_1 far from the data, at the z each model's VUD selected: total uncertainty, uncertainty about z's label,
reaction to z's label and Ve, model vs the exact ControlledGP50-prior reference. Usage: farfield_breakdown.py <results name> ..."""
import sys
import numpy as np, pandas as pd
R = '../results/toy_classification/moons_1'
ref = pd.read_csv('tmp/logs/conditional_reference_moons_1/reference.csv')
cx = pd.read_csv(f'{R}/Dream-v0-Base-7B/va_vanilla/D_test.csv')[['x1', 'x2']].to_numpy(float)
H = lambda p: -(np.clip(p, 1e-12, 1) * np.log2(np.clip(p, 1e-12, 1)) + np.clip(1 - p, 1e-12, 1) * np.log2(np.clip(1 - p, 1e-12, 1)))
print(f'{"model":46s} | TU model/exact | H[p(u|z)] model/exact | response model/exact | Ve model/exact')
for m in sys.argv[1:]:
    run = pd.read_csv(f'{R}/{m}/va_vanilla/results_test.csv')
    sel = run.groupby(['x_x1', 'x_x2'], sort=False).apply(
        lambda g: g.loc[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))].Va.idxmin(), include_groups=False).to_numpy()
    s, r = run.loc[sel], ref.loc[sel]
    away = np.linalg.norm(s[['x_x1', 'x_x2']].to_numpy()[:, None] - cx[None], axis=-1).min(1) >= .35
    v = lambda a: np.asarray(a)[away].mean()
    print(f'{m:46s} | {v(s["H[p(y|x,D)]"]):.2f} / {v(r.bayes_TU):.2f} | {v(H(s["p(u=1|z,D)"].to_numpy())):.2f} / {v(H(r.bayes_pu.to_numpy())):.2f} | '
          f'{v((s["p(y=1|x,u=1,z,D)"] - s["p(y=1|x,u=0,z,D)"]).abs()):.2f} / {v(r.bayes_response):.2f} | '
          f'{v(s["H[p(y|x,D)]"] - s.Va):.2f} / {v(r.bayes_TU - r.bayes_Va):.2f}')
