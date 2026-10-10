"""Mode-1 + ideal-VUD check on moons_1 for the BoTorchOS checkpoint's prior.

Prior: GP (Matern-2.5, ARD lengthscales ~ Gamma(3, 6), outputscale ~ Gamma(2, 0.15),
latent noise 0.001), labels sigmoid -> Bernoulli. Inputs z-scored with the context
(as the MDLM sees them) and mapped to the prior's box as 0.5 + z / sqrt(12).
For each grid x (with the MDLM run's own 15 z): exact Bayes p(y|x,D), its split
(aleatoric E[H(p)], epistemic MI), and VUD's procedure on the exact posterior.
"""
import math
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from bayes_gp_classification import entropy_bits
from evaluate_vud_classification import true_p1
from small_mdlm.episodes import normalize_by_context

torch.set_default_dtype(torch.float64)
dev = 'cuda'
R = '../results/toy_classification/moons_1'
OS = 2.0
H = lambda p: float(entropy_bits(torch.tensor(p)))
bits = lambda p: -(np.clip(p,1e-12,1)*np.log2(np.clip(p,1e-12,1)) + np.clip(1-p,1e-12,1)*np.log2(np.clip(1-p,1e-12,1)))


@torch.no_grad()
def draw(points, y, hyper=4000, funcs=400, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    n, P, d = len(y), len(points), points.shape[1]
    ls = torch._standard_gamma(torch.full((hyper, d), 3.0, device=dev), generator=g) / 6.0
    sc = torch._standard_gamma(torch.full((hyper,), OS, device=dev), generator=g) / 0.15
    pts = torch.tensor(points, device=dev)
    diff = (pts[:, None, :] - pts[None, :, :])[None] / ls[:, None, None, :]
    s = math.sqrt(5) * diff.norm(dim=-1)
    K = sc[:, None, None] * (1 + s + s ** 2 / 3) * torch.exp(-s) + (1e-3 + 1e-8) * torch.eye(P, device=dev)
    f = torch.linalg.cholesky(K) @ torch.randn(hyper, P, funcs, device=dev, generator=g)
    yy = torch.tensor(y, device=dev)[None, :, None]
    logw = (yy * torch.nn.functional.logsigmoid(f[:, :n]) + (1 - yy) * torch.nn.functional.logsigmoid(-f[:, :n])).sum(1).reshape(-1)
    w = torch.exp(logw - logw.max())
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return torch.sigmoid(f[:, n:]).permute(1, 0, 2).reshape(P - n, -1), w / w.sum(), ess


def kl(p, q):
    p, q = np.clip(p, 1e-12, 1 - 1e-12), np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


ctx = pd.read_csv(f'{R}/MDLM-PFNClassify2D-BoTorchOS-0.8M/va_vanilla/D_test.csv')
cx, cy = ctx[['x1', 'x2']].to_numpy(float), ctx['label'].to_numpy(float)
runs = pd.read_csv(f'{R}/MDLM-PFNClassify2D-BoTorchOS-0.8M/va_vanilla/results_test.csv')
rows, esses = [], []
for (x1, x2), g in runs.groupby(['x_x1', 'x_x2']):
    q = np.vstack([[x1, x2], g[['z_x1', 'z_x2']].to_numpy(float)])
    z = normalize_by_context(np.vstack([cx, q]), list(range(len(cx))))
    raw = 0.5 + z / math.sqrt(12)
    p, w, ess = draw(raw, cy)
    esses.append(ess)
    px = p[0]
    p1 = float(px @ w)
    tu, ale = H(p1), float(entropy_bits(px) @ w)
    vas, kls = [], []
    for j in range(1, len(q)):
        pz = p[j]; pu1 = float(pz @ w)
        a, b = float((pz * px) @ w) / pu1, float(((1 - pz) * px) @ w) / (1 - pu1)
        vas.append(pu1 * H(a) + (1 - pu1) * H(b)); kls.append(kl(p1, pu1 * a + (1 - pu1) * b))
    va = min(min(np.array(vas)[np.argsort(kls, kind='stable')[:5]]), tu)
    m = g.iloc[0]
    rows.append((x1, x2, p1, tu, ale, va, m['p(y=1|x,D)'], m['H[p(y|x,D)]'],
                 min(g.nsmallest(5, 'kl_pyx_pyxz')['Va'].min(), m['H[p(y|x,D)]'])))
d = pd.DataFrame(rows, columns=['x1', 'x2', 'bayes_p1', 'bayes_TU', 'bayes_ale', 'ideal_va', 'mdlm_p1', 'mdlm_TU', 'mdlm_va'])
truth = bits(true_p1('moons_1', d[['x1', 'x2']].to_numpy()))
near = np.min(np.linalg.norm(d[['x1', 'x2']].to_numpy()[:, None] - cx[None], axis=-1), 1) < 0.35
print(f'Monte Carlo effective sample size: median {np.median(esses):.0f}, min {np.min(esses):.0f}')
print('Mode 1 (MDLM vs exact Bayes of its prior): mean |p1 diff| '
      f'{np.mean(np.abs(d.mdlm_p1 - d.bayes_p1)):.3f}, mean |total diff| {np.mean(np.abs(d.mdlm_TU - d.bayes_TU)):.3f} bits')
for label, mask in (('near data', near), ('away from data', ~near)):
    s = d[mask]
    print(f'{label:15s} split ale/epi: exact Bayes {s.bayes_ale.mean():.2f}/{(s.bayes_TU - s.bayes_ale).mean():.2f} | '
          f'ideal VUD {s.ideal_va.mean():.2f}/{(s.bayes_TU - s.ideal_va).mean():.2f} | MDLM VUD {s.mdlm_va.mean():.2f}/{(s.mdlm_TU - s.mdlm_va).mean():.2f} | '
          f'true aleatoric {truth[mask].mean():.2f}')
d.to_csv('tmp/logs/mode1_moons1.csv', index=False)
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
g1, g2 = sorted(d.x1.unique()), sorted(d.x2.unique())
grid = lambda col: d.pivot(index='x2', columns='x1', values=col).to_numpy()
d['bayes_epi'] = d.bayes_TU - d.bayes_ale; d['mdlm_epi'] = d.mdlm_TU - d.mdlm_va
for ax, col, title, cmap, vmax in zip(axes.ravel(),
        ['bayes_p1', 'bayes_ale', 'bayes_epi', 'mdlm_p1', 'mdlm_va', 'mdlm_epi'],
        ['Exact Bayes p(y=1)', 'Exact Bayes aleatoric', 'Exact Bayes epistemic', 'MDLM p(y=1)', 'MDLM VUD Va', 'MDLM VUD Ve'],
        ['RdBu_r', 'viridis', 'viridis', 'RdBu_r', 'viridis', 'viridis'], [1, 1, 1, 1, 1, 1]):
    mesh = ax.pcolormesh(g1, g2, grid(col), cmap=cmap, vmin=0, vmax=vmax, shading='auto')
    ax.scatter(cx[:, 0], cx[:, 1], c=cy, cmap='RdBu_r', edgecolors='w', s=18, vmin=0, vmax=1)
    ax.set(title=title); fig.colorbar(mesh, ax=ax)
fig.suptitle('moons_1, BoTorchOS prior: exact Bayes (top) vs MDLM through VUD (bottom), bits')
fig.savefig('tmp/logs/mode1_moons1.png', dpi=110)
