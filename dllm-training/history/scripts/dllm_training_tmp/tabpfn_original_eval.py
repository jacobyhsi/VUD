"""Score the MDLMs on datasets simulated with TabPFN v1's original prior code (tabpfn_original_generate.py).

Each dataset: the first n_context points are the context, the remaining points are test points (TabPFN's split).
Predictions p(y=1 | x, D) for every test point from:
  - MDLM B6 (trained on our port of TabPFN's non-causal MLP prior) and the final MDLM checkpoint;
  - TabPFN v1 itself (released checkpoint, standard TabPFNClassifier inference): trained on exactly this prior, so
    it is the reference for the Bayes posterior predictive (no exact Bayes or true p(y|x) exists for this prior);
  - logistic regression, a GP classifier, and the context base rate.
Metrics: log-loss (bits), accuracy, expected calibration error, TU = mean predictive entropy (bits), and mean
|p - p_TabPFN|; overall and split by the test point's distance from the context (max |x - mean| / std over features).
Usage (from dllm-training/): python tmp/scripts/tabpfn_original_eval.py <datasets.npz> <out.csv> [dataset id offset]
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from small_mdlm.tabpfn_v1 import setup                                       # noqa: E402
REPO = setup()
from tabpfn import TabPFNClassifier                                          # noqa: E402
from sklearn.gaussian_process import GaussianProcessClassifier              # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel            # noqa: E402
from sklearn.linear_model import LogisticRegression                         # noqa: E402

from small_mdlm.backend import NumericMDLM                                  # noqa: E402
from small_mdlm.model import answer_probs                                   # noqa: E402

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OFFSET = int(sys.argv[3]) if len(sys.argv) > 3 else 0
dev = 'cuda'
MODELS = {'MDLM_b6': 'outputs/tabpfn_classify2d/checkpoint.pt',
          'MDLM_final': 'outputs/tabpfn_controlled_gp50_geometry_1d_layout75/checkpoint_step_020000.pt'}
if os.environ.get('EVAL_MODELS'):                                           # e.g. EVAL_MODELS=name=path,name2=path2
    MODELS = dict(item.split('=', 1) for item in os.environ['EVAL_MODELS'].split(','))


@torch.no_grad()
def mdlm_p1(model, cx, cy, qx):
    """p(y=1 | query, context) for each query row, one sequence per query (context + query)."""
    n, tok, out = len(cx), model.tokenizer, []
    pad = lambda a: np.hstack([a, np.zeros((len(a), 2 - a.shape[1]))])     # 1-D tasks use x2 = 0
    enc = [tok.encode(model.prepare_x(pad(np.vstack([cx, q])), [n]), np.append(cy, np.nan), [n], hide=True) for q in qx]
    for s in range(0, len(enc), 512):
        part = enc[s:s + 512]
        ids, kinds, attention, values = tok.collate(part, model.device)
        probs = answer_probs(model.model, tok, ids, kinds, attention, values, 1.0)
        for b, e in enumerate(part):
            (pos,) = e['spans'][n]
            pr = probs[b, pos, 12:12 + tok.labels].double().cpu().numpy()
            out.append(pr[1] / pr.sum())
    return np.array(out)


def safe(fn, cx, cy, qx, fallback):
    try:
        return fn(cx, cy, qx)
    except Exception:
        return np.full(len(qx), fallback)


data = np.load(DATA)
X, Y, NC = data['x'], data['y'], data['n_context']
mdlms = {k: NumericMDLM(v, device=dev) for k, v in MODELS.items()}
tabpfn = TabPFNClassifier(device=dev, base_path=str(REPO/'tabpfn'), N_ensemble_configurations=32)
rows = []
for i in range(len(X)):
    used = np.abs(X[i]).sum(0) > 0                                          # TabPFN zero-pads unused features
    if not used.any():                                                      # degenerate: no informative feature
        continue
    x, y, n = X[i][:, used], Y[i], int(NC[i])
    cx, cy, qx, qy = x[:n], y[:n], x[n:], y[n:]
    mean, std = cx.mean(0), cx.std(0) + 1e-6
    preds = {k: mdlm_p1(m, cx, cy, qx) for k, m in mdlms.items()}
    preds['TabPFN_v1'] = tabpfn.fit(cx, cy.astype(int), overwrite_warning=True).predict_proba(qx)[:, 1]
    preds['logistic_regression'] = safe(lambda a, b, q: LogisticRegression().fit(a, b).predict_proba(q)[:, 1], cx, cy, qx, cy.mean())
    preds['GP_classifier'] = safe(lambda a, b, q: GaussianProcessClassifier(ConstantKernel() * RBF()).fit(a, b).predict_proba(q)[:, 1], cx, cy, qx, cy.mean())
    preds['base_rate'] = np.full(len(qx), (cy.sum() + 1) / (n + 2))
    dist = np.abs((qx - mean) / std).max(1)
    for j in range(len(qx)):
        rows.append({'dataset': OFFSET + i, 'd': int(used.sum()), 'n': n, 'dist': dist[j], 'y': qy[j], **{k: float(v[j]) for k, v in preds.items()}})
    if i % 25 == 0:
        print(f'dataset {i} done', flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
pd.DataFrame(rows).to_csv(OUT, index=False)
print('saved', OUT)
