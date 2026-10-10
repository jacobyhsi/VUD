"""Exact-posterior (ideal VUD) screen of a regression prior on a VUD toy-regression dataset, before training.

Coordinates: x and y are z-scored by the context (how the regression MDLM will see them); entropies are
reported back in the data's units (+ log of the context's y std). Prior members, 50/50:
  het_gp   controlled GP regression: f ~ GP(0, RBF(l)), l ~ logU[0.5, 3]; noise std sigma0 from the controlled
           GP's regimes (clean: variance ratio logU[1e-6, 1e-3], noisy: logU[0.03, 1], equally likely); with
           probability 0.5 heteroskedastic: sigma(x) = sigma0 * exp(a h(x)), a ~ U(0, 3), h ~ GP(0, RBF(l_g)),
           l_g ~ logU[0.5, 3]. f is integrated analytically given (hyperparameters, h).
  mlp      TabPFN v1 MLP prior in regression mode (small_mdlm/vendor/tabpfn_prior.py, non-causal): the network's
           noisy output, standardized over the dataset's 34 points (v1's normalize_data); per-network output
           mean/variance at each point from 32 noise passes (Gaussian approximation).
For every grid x and its 5 VUD z's (from an existing run), weighted posterior draws give p(y|x,D), p(u|z,D) and
p(y|x,z,u,D) (mixtures, moment-matched to Gaussians as VUD does), Va(z) = E_u H[p(y|x,z,u,D)] over 10 u's,
the ideal Va = min over z (capped at TU) and Ve = TU - Va.
Usage (from dllm-training/): python tmp/scripts/regression_reference.py <linear_regression|gaps> [members=het_gp,mlp]
Environment: LS_MIN (lower end of the GP function lengthscale range, default 0.5), GP_HYPER (GP draws, default 400,000),
MLP_CHUNKS (chunks of 2,000 networks, default 200), SEED (offsets both members' seeds, default 0) and TAG (output suffix).
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'tmp/scripts')
from small_mdlm.vendor import tabpfn_prior as tabpfn
from regression_breakdown import true_entropy

torch.set_default_dtype(torch.float64)
dev = 'cuda'
DATASET = sys.argv[1]
MEMBERS = (sys.argv[2] if len(sys.argv) > 2 else 'het_gp,mlp').split(',')
RUN = f'../results/toy_regression/{DATASET}/MDLM-BayesRegressGP-0.8M/va_vanilla/seed_0'
KEEP, N_U, SEQ = 4000, 10, 34
LS_MIN = float(os.environ.get('LS_MIN', 0.5))
GP_HYPER = int(os.environ.get('GP_HYPER', 400_000))
MLP_CHUNKS = int(os.environ.get('MLP_CHUNKS', 200))
SEED = int(os.environ.get('SEED', 0))
LOG2PIE = math.log(2 * math.pi * math.e)


def log_u(n, lo, hi, g):
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(n, device=dev, generator=g))


def rbf(a, b, ls):
    return torch.exp(-0.5 * (a[:, :, None] - b[:, None]) ** 2 / ls[:, None, None] ** 2)


def resample(w, keep, g):
    cdf = torch.cumsum(w, 0)
    return torch.searchsorted(cdf, torch.rand(keep, device=dev, generator=g) * cdf[-1]).clamp(max=len(w) - 1)


@torch.no_grad()
def het_gp_member(cx, cy, hyper=GP_HYPER, chunk=20_000, seed=SEED):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(cx)
    X, Y = torch.tensor(cx, device=dev), torch.tensor(cy, device=dev)
    keep_parts = {k: [] for k in ('ls', 'lsg', 's0', 'a', 'h', 'logw')}
    for _ in range(hyper // chunk):
        ls, lsg = log_u(chunk, LS_MIN, 3, g), log_u(chunk, .5, 3, g)
        clean = torch.rand(chunk, device=dev, generator=g) < .5
        s0 = torch.sqrt(torch.where(clean, log_u(chunk, 1e-6, 1e-3, g), log_u(chunk, .03, 1, g)))
        a = torch.where(torch.rand(chunk, device=dev, generator=g) < .5, 3 * torch.rand(chunk, device=dev, generator=g), torch.zeros(chunk, device=dev))
        Xe = X.expand(chunk, n)
        Kg = rbf(Xe, Xe, lsg) + 1e-6 * torch.eye(n, device=dev)
        h = (torch.linalg.cholesky(Kg) @ torch.randn(chunk, n, 1, device=dev, generator=g)).squeeze(-1)
        sig2 = (s0[:, None] * torch.exp(a[:, None] * h)) ** 2
        K = rbf(Xe, Xe, ls) + torch.diag_embed(sig2) + 1e-8 * torch.eye(n, device=dev)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(Y.expand(chunk, n)[..., None], L).squeeze(-1)
        logw = -0.5 * (Y * alpha).sum(-1) - torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1) - 0.5 * n * math.log(2 * math.pi)
        for k, v in zip(keep_parts, (ls, lsg, s0, a, h, logw)):
            keep_parts[k].append(v)
    ls, lsg, s0, a, h, logw = (torch.cat(keep_parts[k]) for k in keep_parts)
    w = torch.exp(logw - logw.max())
    idx = resample(w, KEEP, g)
    return {'log_z': float(torch.logsumexp(logw, 0) - math.log(len(logw))), 'ess': float(w.sum() ** 2 / (w ** 2).sum()),
            'ls': ls[idx], 'lsg': lsg[idx], 's0': s0[idx], 'a': a[idx], 'h': h[idx], 'X': X, 'Y': Y}


@torch.no_grad()
def het_gp_joint(m, q, g):
    """Per draw: mean (S, Q) and covariance (S, Q, Q) of the noisy y at query points q (Q,)."""
    S, n = len(m['ls']), len(m['X'])
    Xe, Q = m['X'].expand(S, n), torch.tensor(q, device=dev).expand(S, len(q))
    Kg = rbf(Xe, Xe, m['lsg']) + 1e-6 * torch.eye(n, device=dev)
    Lg = torch.linalg.cholesky(Kg)
    kgq = rbf(Q, Xe, m['lsg'])
    vg = torch.linalg.solve_triangular(Lg, kgq.transpose(1, 2), upper=False)
    hq_mean = (kgq @ torch.cholesky_solve(m['h'][..., None], Lg)).squeeze(-1)
    hq_cov = rbf(Q, Q, m['lsg']) - vg.transpose(1, 2) @ vg + 1e-6 * torch.eye(len(q), device=dev)
    hq = hq_mean + (torch.linalg.cholesky(hq_cov) @ torch.randn(S, len(q), 1, device=dev, generator=g)).squeeze(-1)
    sig2_c = (m['s0'][:, None] * torch.exp(m['a'][:, None] * m['h'])) ** 2
    sig2_q = (m['s0'][:, None] * torch.exp(m['a'][:, None] * hq)) ** 2
    K = rbf(Xe, Xe, m['ls']) + torch.diag_embed(sig2_c) + 1e-8 * torch.eye(n, device=dev)
    L = torch.linalg.cholesky(K)
    kq = rbf(Q, Xe, m['ls'])
    mean = (kq @ torch.cholesky_solve(m['Y'].expand(S, n)[..., None], L)).squeeze(-1)
    v = torch.linalg.solve_triangular(L, kq.transpose(1, 2), upper=False)
    cov = rbf(Q, Q, m['ls']) - v.transpose(1, 2) @ v + torch.diag_embed(sig2_q)
    return mean, cov


def mlp_chunk(seed, cx, chunk=2000, draws=32):
    g = torch.Generator(device=dev).manual_seed(seed)
    net = tabpfn.sample_networks(chunk, 1, g, dev)
    xc = net['x_mean'][:, None] + net['x_std'][:, None] * cx[None, :, None]
    pts = torch.cat([xc, tabpfn.sample_inputs(net, SEQ - len(cx), g)], 1)
    once = tabpfn.forward(net, pts, g)                                   # one dataset realisation, for v1's y normalisation
    mu, sd = once.mean(1, keepdim=True), once.std(1, keepdim=True) + 1e-6
    outs = torch.stack([tabpfn.forward(net, xc, g) for _ in range(draws)], 0)
    return net, mu, sd, ((outs - mu) / sd).mean(0), ((outs - mu) / sd).var(0) + 1e-4


@torch.no_grad()
def mlp_member(cx, cy, chunks=MLP_CHUNKS, seed=SEED + 1):
    torch.set_default_dtype(torch.float32)
    X, Y = torch.tensor(cx, device=dev, dtype=torch.float32), torch.tensor(cy, device=dev, dtype=torch.float32)
    logw = []
    for c in range(chunks):
        _, _, _, m, v = mlp_chunk(seed * 100_003 + c, X)
        lw = (-0.5 * (Y - m) ** 2 / v - 0.5 * torch.log(2 * math.pi * v)).sum(-1).double()
        logw.append(torch.nan_to_num(lw, nan=-math.inf))
    logw = torch.cat(logw)
    w = torch.exp(logw - logw.max())
    idx = resample(w, KEEP, torch.Generator(device=dev).manual_seed(seed)).sort().values
    nets, mus, sds = [], [], []
    for c in idx.div(2000, rounding_mode='floor').unique().tolist():
        net, mu, sd, _, _ = mlp_chunk(seed * 100_003 + c, X)
        local = idx[(idx >= c * 2000) & (idx < (c + 1) * 2000)] - c * 2000
        nets.append({k: v[local] for k, v in net.items()}); mus.append(mu[local]); sds.append(sd[local])
    torch.set_default_dtype(torch.float64)
    return {'log_z': float(torch.logsumexp(logw, 0) - math.log(len(logw))), 'ess': float(w.sum() ** 2 / (w ** 2).sum()),
            'net': {k: torch.cat([n_[k] for n_ in nets]) for k in nets[0]}, 'mu': torch.cat(mus), 'sd': torch.cat(sds)}


@torch.no_grad()
def mlp_joint(m, q, g, draws=32):
    """Per network: independent Gaussian outputs at q (no covariance between points given the network)."""
    qq = torch.tensor(q, device=dev, dtype=torch.float32)
    net = m['net']
    xq = (net['x_mean'][:, None] + net['x_std'][:, None] * qq[None, :, None]).contiguous()
    outs = torch.stack([tabpfn.forward(net, xq, g) for _ in range(draws)], 0)
    z = (outs - m['mu']) / m['sd']
    return z.mean(0).double(), torch.diag_embed(z.var(0).double() + 1e-4)


def gauss_h(var):
    return 0.5 * (LOG2PIE + np.log(np.maximum(var, 1e-300)))


def vud_from_draws(mean, cov, w, rng):
    """mean (S, 6), cov (S, 6, 6), weights w (S,): row 0 = x, rows 1.. = the z's. Returns TU and Va per z (normalized units)."""
    mx, vx = mean[:, 0], cov[:, 0, 0]
    tot_mean = (w * mx).sum(); tu = gauss_h((w * (vx + mx ** 2)).sum() - tot_mean ** 2)
    vas = []
    for j in range(1, mean.shape[1]):
        mz, vz, c = mean[:, j], cov[:, j, j], cov[:, 0, j]
        pick = rng.choice(len(w), size=N_U, p=w)
        us = mz[pick] + np.sqrt(vz[pick]) * rng.standard_normal(N_U)
        hs = []
        for u in us:
            lw = np.log(w + 1e-300) - 0.5 * (u - mz) ** 2 / vz - 0.5 * np.log(vz)
            wu = np.exp(lw - lw.max()); wu /= wu.sum()
            m_c, v_c = mx + c / vz * (u - mz), vx - c ** 2 / vz
            mu_ = (wu * m_c).sum()
            hs.append(gauss_h((wu * (v_c + m_c ** 2)).sum() - mu_ ** 2))
        vas.append(np.mean(hs))
    return tu, np.array(vas)


ctx = pd.read_csv(f'{RUN}/D_test.csv')
cx_raw, cy_raw = ctx.iloc[:, 0].to_numpy(float), ctx.iloc[:, 1].to_numpy(float)
xm, xs, ym, ys = cx_raw.mean(), cx_raw.std(), cy_raw.mean(), cy_raw.std()
cx, cy = (cx_raw - xm) / xs, (cy_raw - ym) / ys
members = {}
if 'het_gp' in MEMBERS:
    members['het_gp'] = het_gp_member(cx, cy)
if 'mlp' in MEMBERS:
    members['mlp'] = mlp_member(cx, cy)
logz = {k: math.log(1 / len(members)) + m['log_z'] for k, m in members.items()}
top = max(logz.values())
post = {k: math.exp(v - top) for k, v in logz.items()}
post = {k: v / sum(post.values()) for k, v in post.items()}
print(DATASET, {k: {'log10_evidence': round(m['log_z'] / math.log(10), 3), 'ess': round(m['ess']), 'posterior_weight': round(post[k], 3)}
                for k, m in members.items()}, flush=True)

res = pd.read_csv(f'{RUN}/results_test.csv')
xcol = [c for c in res.columns if c.startswith('x_') and c != 'x_note'][0]
zcol = [c for c in res.columns if c.startswith('z_') and c != 'z_note'][0]
rng = np.random.default_rng(0)
rows = []
for i, (xv, grp) in enumerate(res.groupby(xcol, sort=True)):
    q = (np.append(xv, grp[zcol].to_numpy(float)) - xm) / xs
    g = torch.Generator(device=dev).manual_seed(i)
    means, covs, ws = [], [], []
    for k, m in members.items():
        mean, cov = het_gp_joint(m, q, g) if k == 'het_gp' else mlp_joint(m, q, g)
        means.append(mean.cpu().numpy()); covs.append(cov.cpu().numpy()); ws.append(np.full(len(mean), post[k] / len(mean)))
    mean, cov, w = np.concatenate(means), np.concatenate(covs), np.concatenate(ws)
    tu, vas = vud_from_draws(mean, cov, w / w.sum(), rng)
    rows.append((xv, tu + np.log(ys), min(vas.min(), tu) + np.log(ys)))
d = pd.DataFrame(rows, columns=['x', 'TU', 'Va'])
d['true'] = true_entropy(DATASET, d.x.to_numpy())
tag = '+'.join(members) + (f'_ls{LS_MIN:g}' if LS_MIN != 0.5 else '') + os.environ.get('TAG', '')
d.to_csv(f'tmp/logs/regression_reference_{DATASET}_{tag}.csv', index=False)
dist = np.abs(d.x.to_numpy()[:, None] - cx_raw[None]).min(1)
inside = (d.x >= cx_raw.min()) & (d.x <= cx_raw.max())
regions = {'on data': dist <= .5, 'gaps': np.asarray(inside & (dist > 1)), 'beyond': np.asarray(~inside & (dist > 1))}
if DATASET == 'gaps':
    regions = {'tight clusters (x~-7, -1)': (dist <= .5) & (d.x < 2).to_numpy(), 'noisy cluster (x~5)': (dist <= .5) & (d.x >= 2).to_numpy(),
               'gaps': regions['gaps'], 'beyond': regions['beyond']}
print(f'ideal VUD under [{tag}] (nats): region | TU | Va (true) | |Va-true| | Ve')
for name, mask in regions.items():
    if mask.any():
        s = d[mask]
        print(f'   {name:26s} | {s.TU.mean():5.2f} | {s.Va.mean():5.2f} ({s.true.mean():5.2f}) | {np.abs(s.Va - s.true).mean():5.2f} | {(s.TU - s.Va).mean():.2f}')
