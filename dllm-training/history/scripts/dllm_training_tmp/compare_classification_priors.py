"""VUD Va/Ve vs the true aleatoric on logistic_regression and moons_1 for the PFN-prior checkpoints and Dream."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1
bits = lambda p: -(np.clip(p,1e-12,1)*np.log2(np.clip(p,1e-12,1)) + np.clip(1-p,1e-12,1)*np.log2(np.clip(1-p,1e-12,1)))
MODELS = ['MDLM-BayesClassify2D-0.8M', 'MDLM-BayesClassify2D-2.5M', 'MDLM-PFNClassify2D-0.8M', 'MDLM-PFNClassify2D-BoTorchOS-0.8M', 'MDLM-PFNClassify2D-MixedLabels-0.8M', 'MDLM-TabPFNClassify2D-0.8M', 'MDLM-TabPFNClassify2D-0.8M-60k', 'MDLM-TabPFNClassify2D-2.7M', 'MDLM-TabPFNClassify2D-IIDControl-0.8M', 'MDLM-TabPFNClassify2D-Geometry-0.8M', 'MDLM-TabPFNFullClassify2D-0.8M', 'MDLM-TabPFN-GPMLP50-0.8M', 'MDLM-TabPFN-GPMLP96-0.8M', 'MDLM-TabPFN-ControlledGP50-0.8M', 'Dream-v0-Base-7B']

def decomposition(dataset, model):
    try:
        df = pd.read_csv(f'results/toy_classification/{dataset}/{model}/va_vanilla/results_test.csv')
    except FileNotFoundError:
        return None
    cols = [c for c in df.columns if c.startswith('x_x')]
    rows = [(*np.atleast_1d(k), g['H[p(y|x,D)]'].iloc[0], min(g.nsmallest(5, 'kl_pyx_pyxz')['Va'].min(), g['H[p(y|x,D)]'].iloc[0]))
            for k, g in df.groupby(cols)]
    return pd.DataFrame(rows, columns=cols + ['TU', 'Va'])

for dataset in ('logistic_regression', 'moons_1'):
    ctx = pd.read_csv(f'results/toy_classification/{dataset}/Dream-v0-Base-7B/va_vanilla/D_test.csv')
    cx = ctx[[c for c in ctx.columns if c.startswith('x')]].to_numpy(float)
    print(f'== {dataset}: |Va - true aleatoric| and split Va/Ve (bits)')
    for model in MODELS:
        d = decomposition(dataset, model)
        if d is None:
            print(f'   {model:38s} (no VUD run)'); continue
        x = d[[c for c in d.columns if c.startswith('x_x')]].to_numpy(float)
        truth = bits(true_p1(dataset, x))
        if dataset == 'logistic_regression':
            near = (x[:, 0] >= cx.min()) & (x[:, 0] <= cx.max())
        else:
            near = np.min(np.linalg.norm(x[:, None] - cx[None], axis=-1), 1) < 0.35
        err = np.abs(d.Va - truth); ve = d.TU - d.Va
        print(f'   {model:38s} near {err[near].mean():.3f} ({d.Va[near].mean():.2f}/{ve[near].mean():.2f})  '
              f'away {err[~near].mean():.3f} ({d.Va[~near].mean():.2f}/{ve[~near].mean():.2f})  all {err.mean():.3f}')
    print(f'   true aleatoric mean: near {truth[near].mean():.2f}, away {truth[~near].mean():.2f}')
