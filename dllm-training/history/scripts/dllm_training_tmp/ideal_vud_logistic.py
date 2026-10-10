"""VUD vanilla applied to the exact Bayes posterior of each PFN GP prior (logistic_regression).

For every grid x, the 15 auxiliary z come from the MDLM run's own results. Latent
functions are drawn jointly at the context, x and its z's (Monte Carlo over the
GP prior, weighted by the context likelihood), giving exact:
  p(y|x,D), p(u|z,D), p(y|x,u,z,D)  ->  Va(z) = sum_u p(u|z,D) H[p(y|x,u,z,D)]
then VUD's rule: min Va among the 5 lowest-KL z, capped at total. Bits.
Compared with the exact Bayes split (aleatoric E[H(p)], epistemic MI) and
with VUD run on the MDLM.
"""
import math
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
from bayes_gp_classification import entropy_bits, matern52
from evaluate_vud_classification import true_p1
from small_mdlm.episodes import normalize_by_context

R = '../results/toy_classification/logistic_regression'
torch.set_default_dtype(torch.float64)
dev = 'cuda'


def to_raw(values, context):
    z = normalize_by_context(np.concatenate([context, values])[:, None], list(range(len(context))))[:, 0]
    return 0.5 + z[len(context):] / math.sqrt(12), 0.5 + z[:len(context)] / math.sqrt(12)


@torch.no_grad()
def draw(points, y, os, hyper=3000, funcs=300, seed=0):
    """Weighted latent draws at points (context first): returns sigmoid(f) at non-context points and weights."""
    g = torch.Generator(device=dev).manual_seed(seed)
    n, P = len(y), len(points)
    ls = torch._standard_gamma(torch.full((hyper,), 3.0, device=dev), generator=g) / 6.0
    sc = torch._standard_gamma(torch.full((hyper,), os, device=dev), generator=g) / 0.15
    pts = torch.tensor(points, device=dev)[:, None]
    r = (pts - pts.T).abs()[None] / ls[:, None, None]
    s = math.sqrt(5) * r
    K = sc[:, None, None] * (1 + s + s ** 2 / 3) * torch.exp(-s) + (1e-3 + 1e-8) * torch.eye(P, device=dev)
    f = torch.linalg.cholesky(K) @ torch.randn(hyper, P, funcs, device=dev, generator=g)
    yy = torch.tensor(y, device=dev)[None, :, None]
    logw = (yy * torch.nn.functional.logsigmoid(f[:, :n]) + (1 - yy) * torch.nn.functional.logsigmoid(-f[:, :n])).sum(1)
    w = torch.exp(logw - logw.max()).reshape(-1)
    return torch.sigmoid(f[:, n:]).permute(1, 0, 2).reshape(P - n, -1), w / w.sum()


def bernoulli_kl(p, q):
    p, q = np.clip(p, 1e-12, 1 - 1e-12), np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


ctx = pd.read_csv(f'{R}/MDLM-PFNClassify2D-0.8M/va_vanilla/D_test.csv')
cx, cy = ctx['x1'].to_numpy(), ctx['label'].to_numpy().astype(float)
H = lambda p: float(entropy_bits(torch.tensor(p)))
bits = lambda p: -(np.clip(p,1e-12,1)*np.log2(np.clip(p,1e-12,1)) + np.clip(1-p,1e-12,1)*np.log2(np.clip(1-p,1e-12,1)))
for os, folder in ((0.5, 'MDLM-PFNClassify2D-0.8M'), (2.0, 'MDLM-PFNClassify2D-BoTorchOS-0.8M')):
    runs = pd.read_csv(f'{R}/{folder}/va_vanilla/results_test.csv')
    rows = []
    for x, g in runs.groupby('x_x1'):
        zs = g['z_x1'].to_numpy()
        raw_q, raw_c = to_raw(np.concatenate([[x], zs]), cx)
        p, w = draw(np.concatenate([raw_c, raw_q]), cy, os)
        px = p[0]
        p1 = float(px @ w)
        tu = H(p1)
        aleatoric = float(entropy_bits(px) @ w)
        vas, kls = [], []
        for j in range(len(zs)):
            pz = p[1 + j]
            pu1 = float(pz @ w)
            py_u1 = float((pz * px) @ w) / pu1
            py_u0 = float(((1 - pz) * px) @ w) / (1 - pu1)
            vas.append(pu1 * H(py_u1) + (1 - pu1) * H(py_u0))
            kls.append(bernoulli_kl(p1, pu1 * py_u1 + (1 - pu1) * py_u0))
        order = np.argsort(kls, kind='stable')[:5]
        va = min(min(np.array(vas)[order]), tu)
        m = g.iloc[0]
        rows.append((x, tu, aleatoric, va, float(np.mean(vas)), m['H[p(y|x,D)]'],
                     min(g.nsmallest(5, 'kl_pyx_pyxz')['Va'].min(), m['H[p(y|x,D)]'])))
    d = pd.DataFrame(rows, columns=['x', 'TU', 'bayes_ale', 'ideal_va', 'ideal_va_mean_z', 'mdlm_TU', 'mdlm_va'])
    truth = bits(true_p1('logistic_regression', d[['x']].to_numpy()))
    inside = (d.x >= cx.min()) & (d.x <= cx.max())
    print(f'== prior outputscale Gamma({os}, 0.15)   (aleatoric / epistemic, bits)')
    for label, mask in (('inside', inside), ('outside', ~inside)):
        s = d[mask]
        print(f'   {label:7s} exact Bayes {s.bayes_ale.mean():.2f}/{(s.TU-s.bayes_ale).mean():.2f} | '
              f'ideal VUD {s.ideal_va.mean():.2f}/{(s.TU-s.ideal_va).mean():.2f} (avg over z {s.ideal_va_mean_z.mean():.2f}) | '
              f'MDLM VUD {s.mdlm_va.mean():.2f}/{(s.mdlm_TU-s.mdlm_va).mean():.2f} | '
              f'|Va-true|: ideal VUD {np.abs(s.ideal_va-truth[mask]).mean():.3f}, MDLM VUD {np.abs(s.mdlm_va-truth[mask]).mean():.3f}')
    d.to_csv(f'tmp/logs/ideal_vud_logistic_os{os}.csv', index=False)
