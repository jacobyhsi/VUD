"""Mode-1 check: a TabPFN-prior MDLM's p(y|x,D) against the exact Bayes answer of its prior.

Uses the exact answers saved by mode1_bag.py (bag 'TabPFN alone') on the logistic_regression and
moons_1 grids, with the same contexts, and reports the mean |p1 difference| and total-entropy
difference (bits) near and away from the data.
Usage (from dllm-training/): python tmp/scripts/mode1_mdlm_tabpfn.py <run under outputs/> [...]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from small_mdlm.backend import NumericMDLM

R = '../results/toy_classification'
EXACT = {'logistic_regression': ('tmp/logs/mode1_bag_logistic_regression.csv', ['x1']),
         'moons_1': ('tmp/logs/mode1_bag_moons_1_tabpfn2500.csv', ['x1', 'x2'])}
bits = lambda p: -(np.clip(p, 1e-12, 1) * np.log2(np.clip(p, 1e-12, 1)) + np.clip(1 - p, 1e-12, 1) * np.log2(np.clip(1 - p, 1e-12, 1)))

for run in sys.argv[1:]:
    model = NumericMDLM(f'outputs/{run}/checkpoint.pt', device='cuda')
    for dataset, (path, cols) in EXACT.items():
        exact = pd.read_csv(path)
        exact = exact[exact.bag == 'TabPFN alone']
        ctx = pd.read_csv(f'{R}/{dataset}/Dream-v0-Base-7B/va_vanilla/D_test.csv')
        cx, cy = ctx[cols].to_numpy(float), ctx['label'].to_numpy(float)
        pad = lambda a: np.hstack([a, np.zeros((len(a), 2 - a.shape[1]))])
        p1 = np.array([model.label_probs(pad(np.vstack([cx, q[None]])), np.append(cy, 0.0), len(cx))[1]
                       for q in exact[cols].to_numpy(float)])
        x = exact[cols].to_numpy(float)
        near = ((x[:, 0] >= cx.min()) & (x[:, 0] <= cx.max()) if dataset == 'logistic_regression'
                else np.min(np.linalg.norm(x[:, None] - cx[None], axis=-1), 1) < 0.35)
        dp, dtu = np.abs(p1 - exact.p1.to_numpy()), bits(p1) - exact.TU.to_numpy()
        print(f'{run:28s} {dataset:20s} |p1 diff| near {dp[near].mean():.3f} away {dp[~near].mean():.3f} | '
              f'total MDLM-Bayes near {dtu[near].mean():+.3f} away {dtu[~near].mean():+.3f} bits', flush=True)
