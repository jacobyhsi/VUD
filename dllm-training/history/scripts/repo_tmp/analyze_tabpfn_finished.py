import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1

models = ['MDLM-TabPFNClassify2D-0.8M', 'MDLM-PFNClassify2D-BoTorchOS-0.8M', 'MDLM-PFNClassify2D-MixedLabels-0.8M', 'MDLM-BayesClassify2D-2.5M', 'Dream-v0-Base-7B']
for dataset in ['logistic_regression', 'moons_1']:
    root = Path('results/toy_classification') / dataset
    context = pd.read_csv(root / 'Dream-v0-Base-7B/va_vanilla/D_test.csv')
    cols = [c for c in context if c.startswith('x')]
    cx = context[cols].to_numpy()
    print(dataset, flush=True)
    for model in models:
        folder = root / model / 'va_vanilla'
        df = pd.read_csv(folder / 'results_test.csv')
        rows = []
        for key, g in df.groupby(['x_' + c for c in cols]):
            h = g['H[p(y|x,D)]'].iloc[0]
            kl = g.kl_pyx_pyxz
            va = min(g.loc[kl.isin(kl.nsmallest(5)), 'Va'].min(), h)
            rows.append((*np.atleast_1d(key), h, va))
        a = np.array(rows)
        x, h, va = a[:, :-2], a[:, -2], a[:, -1]
        p = np.clip(true_p1(dataset, x), 1e-12, 1-1e-12)
        truth = -p*np.log2(p)-(1-p)*np.log2(1-p)
        near = ((x[:,0] >= cx.min()) & (x[:,0] <= cx.max())) if dataset == 'logistic_regression' else np.min(np.linalg.norm(x[:,None]-cx[None],axis=-1),axis=1)<.35
        errors = [float(np.mean(np.abs(va[mask]-truth[mask]))) for mask in [near, ~near]]
        same = pd.read_csv(folder/'D_test.csv').equals(context)
        print(model, 'points', len(a), 'same_context',same,'near/away_MAE',np.round(errors,4),'mean_Ve',round(float((h-va).mean()),4),flush=True)
