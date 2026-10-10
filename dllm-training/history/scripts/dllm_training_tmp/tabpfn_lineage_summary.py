"""One table for every TabPFN-lineage classification MDLM (plus references): VUD Va error and Ve away from the
data, the response to the auxiliary label away from the data (vs the exact ControlledGP50-prior reference),
and 40-context predictive accuracy / log-loss (moons_2 is held out from all prior choices). Rerun to add runs."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1
RUNS = [  # (label, results name, outputs dir)
    ('TabPFN MLP (non-causal)', 'MDLM-TabPFNClassify2D-0.8M', 'tabpfn_classify2d'),
    ('  + 60k steps', 'MDLM-TabPFNClassify2D-0.8M-60k', 'tabpfn_classify2d_long'),
    ('  + 2.7M params', 'MDLM-TabPFNClassify2D-2.7M', 'tabpfn_classify2d_big'),
    ('  + geometry episodes', 'MDLM-TabPFNClassify2D-Geometry-0.8M', 'tabpfn_classify2d_geometry'),
    ('  + IID-control episodes', 'MDLM-TabPFNClassify2D-IIDControl-0.8M', 'tabpfn_classify2d_iid_control'),
    ('TabPFN MLP causal+non-causal', 'MDLM-TabPFNFullClassify2D-0.8M', 'tabpfn_full_classify2d'),
    ('TabPFN GP + MLP 50/50', 'MDLM-TabPFN-GPMLP50-0.8M', 'tabpfn_gpmlp50_classify2d'),
    ('TabPFN GP + MLP 4/96', 'MDLM-TabPFN-GPMLP96-0.8M', 'tabpfn_gpmlp96_classify2d'),
    ('Controlled GP + MLP 50/50', 'MDLM-TabPFN-ControlledGP50-0.8M', 'tabpfn_controlled_gp50_classify2d'),
    ('  + geometry episodes', 'MDLM-TabPFN-ControlledGP50-Geometry-0.8M', 'tabpfn_controlled_gp50_geometry'),
    ('  + IID-control episodes', 'MDLM-TabPFN-ControlledGP50-IIDControl-0.8M', 'tabpfn_controlled_gp50_iid_control'),
    ('  + more geometry (round 2A)', 'MDLM-TabPFN-ControlledGP50-GeometryFull-0.8M', 'tabpfn_controlled_gp50_geometry_full'),
    ('  + more 1-D tasks (round 2B)', 'MDLM-TabPFN-ControlledGP50-Geometry1D-0.8M', 'tabpfn_controlled_gp50_geometry_1d'),
    ('  + 75% GP prior (round 3A)', 'MDLM-TabPFN-ControlledGP75-Geometry1D-0.8M', 'tabpfn_controlled_gp75_geometry_1d'),
    ('  + 2.7M params (round 3B)', 'MDLM-TabPFN-ControlledGP50-Geometry1D-2.7M', 'tabpfn_controlled_gp50_geometry_1d_big'),
    ('  + input layouts (round 4)', 'MDLM-TabPFN-ControlledGP50-Geometry1D-Layout-0.8M', 'tabpfn_controlled_gp50_geometry_1d_layout'),
    ('  + 40k steps (round 5A)', 'MDLM-TabPFN-ControlledGP50-Geometry1D-Layout-40k-0.8M', 'tabpfn_controlled_gp50_geometry_1d_layout_long'),
    ('  + 75% layouts (round 5B)', 'MDLM-TabPFN-ControlledGP50-Geometry1D-Layout75-0.8M', 'tabpfn_controlled_gp50_geometry_1d_layout75'),
    ('ref: BayesClassify 2.5M (hand-built)', 'MDLM-BayesClassify2D-2.5M', 'bayes_classify2d'),
    ('ref: Dream-v0-Base-7B', 'Dream-v0-Base-7B', None)]
bits = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def vud(ds, name):
    f = Path(f'results/toy_classification/{ds}/{name}/va_vanilla/results_test.csv')
    if not f.exists():
        return None
    cols = ['x1'] if ds == 'logistic_regression' else ['x1', 'x2']
    df = pd.read_csv(f)
    cx = pd.read_csv(f.parent/'D_test.csv')[cols].to_numpy(float)
    rows = []
    for key, g in df.groupby([f'x_{c}' for c in cols]):
        cand = g.loc[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))]
        s = cand.loc[cand.Va.idxmin()]
        h = g['H[p(y|x,D)]'].iloc[0]
        rows.append([*np.atleast_1d(key), h, s.Va if len(cols) == 2 else min(s.Va, h),
                     abs(s['p(y=1|x,u=1,z,D)'] - s['p(y=1|x,u=0,z,D)'])])
    a = np.array(rows); x = a[:, :len(cols)]; tu, va, resp = a[:, len(cols):].T
    ale = bits(np.clip(true_p1(ds, x), 1e-9, 1 - 1e-9))
    near = ((x[:, 0] >= cx[:, 0].min()) & (x[:, 0] <= cx[:, 0].max())) if len(cols) == 1 else np.linalg.norm(x[:, None] - cx[None], axis=-1).min(1) < .35
    err = np.abs(va - ale)
    return {'err_near': err[near].mean(), 'err_away': err[~near].mean(), 've_away': (tu - va)[~near].mean(), 'resp_away': resp[~near].mean()}


def predictive(out, ds):
    f = Path(f'dllm-training/outputs/{out}/evaluation_vud_datasets/summary.json') if out else None
    if not f or not f.exists():
        return None
    return json.loads(f.read_text())['datasets'][ds]['scores']['mdlm']


ref = {}
for ds in ('moons_1', 'logistic_regression'):
    s = Path(f'dllm-training/tmp/logs/conditional_reference_{ds}/summary.json')
    if s.exists():
        r = json.loads(s.read_text())['MDLM-TabPFN-ControlledGP50-0.8M']['away']
        ref[ds] = (r['bayes_response_selected_z'], r['bayes_Ve_at_selected_z'])
fmt = lambda v, k='{:.2f}': '  –  ' if v is None else k.format(v)
print('| Model | moons_1 Va err near / away | moons_1 Ve away | moons_1 response away | LR Va err near / away | LR Ve away | LR response away | moons_1 acc / log-loss | moons_2 acc / log-loss (held out) |')
print('|---|---|---|---|---|---|---|---|---|')
for label, name, out in RUNS:
    m, l = vud('moons_1', name), vud('logistic_regression', name)
    p1, p2 = predictive(out, 'moons_1'), predictive(out, 'moons_2')
    if m is None and l is None and p1 is None:
        print(f'| {label} | (pending) | | | | | | | |'); continue
    c = lambda d, k: None if d is None else d[k]
    print(f'| {label} | {fmt(c(m,"err_near"))} / {fmt(c(m,"err_away"))} | {fmt(c(m,"ve_away"))} | {fmt(c(m,"resp_away"))} | '
          f'{fmt(c(l,"err_near"))} / {fmt(c(l,"err_away"))} | {fmt(c(l,"ve_away"))} | {fmt(c(l,"resp_away"))} | '
          f'{fmt(c(p1,"accuracy"), "{:.3f}")} / {fmt(c(p1,"log_loss"), "{:.3f}")} | {fmt(c(p2,"accuracy"), "{:.3f}")} / {fmt(c(p2,"log_loss"), "{:.3f}")} |')
if ref:
    print(f'| exact answer, ControlledGP50 prior | | {ref["moons_1"][1]:.2f} | {ref["moons_1"][0]:.2f} | | {ref["logistic_regression"][1]:.2f} | {ref["logistic_regression"][0]:.2f} | | |')
