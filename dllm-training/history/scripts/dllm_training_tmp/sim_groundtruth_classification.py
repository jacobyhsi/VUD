"""Simulated ground-truth comparison for the classification MDLM (the professor's check).

PRIOR=tabpfn (default): fresh test tasks (seed 777, never used in training) from TabPFN v1's default classification
generation as released (small_mdlm/vendor/tabpfn_prior.py: network/noise hyperpriors from the released checkpoint,
IID inputs from each network's input law, rank/value two-class threshold), non-causal branch only (only there is the
true p(y|x) known), binary, 1-2 features (50/50), context size 10-30. The exact Bayes reference is this prior alone.
PRIOR=mix: the final checkpoint's own training prior instead (half controlled GP, half TabPFN MLP, 75% layouts).
MODELS: the original TabPFN checkpoint (B6, trained on exactly the TabPFN prior) and the final checkpoint. Each task has 2 "near" queries (unused points of the same dataset) and 2 "far" queries (3-4 context
standard deviations out). For every query we know the TRUE p(y=1|x) (the latent function / network).
  MDLM + VUD: VUD vanilla exactly as src/experiments/toy_classification.py: 15 z ~ N(x, (0.1 * context std)^2),
    Va(z) = sum_u p(u|z,D) H[p(y|x,z,u,D)], min Va among the 5 lowest KL(p(y|x,D) || p(y|x,z,D)), capped at TU.
  Exact Bayes under the same prior (both members, weighted by their evidence; importance sampling): TU, the
    Bayes aleatoric E[H(p)] and epistemic TU - E[H(p)], and VUD run on the exact posterior (same z's).
Approximations (as in conditional_reference.py): the context is z-scored as the MDLM sees it and mapped to each
member's coordinates (GP: directly; MLP: each network's input law); MLP probabilities use 32/64 noise passes.
Usage (from dllm-training/): python tmp/scripts/sim_groundtruth_classification.py <first task> <last task, exclusive>
Writes tmp/logs/sim_groundtruth_classification/queries_<first>_<last>.csv (one row per query).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
from small_mdlm.backend import NumericMDLM
from small_mdlm.episodes import normalize_by_context
from small_mdlm.model import answer_probs
from small_mdlm.tabpfn_episodes import _layout_points
from small_mdlm.vendor import tabpfn_prior as tabpfn
from small_mdlm.vendor.controlled_gp_prior import ControlledGPConfig, sample_at_inputs, sample_hyperparameters

dev = 'cuda'
PRIOR = os.environ.get('PRIOR', 'tabpfn')
MODELS = {'b6': 'outputs/tabpfn_classify2d/checkpoint.pt',
          'final': 'outputs/tabpfn_controlled_gp50_geometry_1d_layout75/checkpoint_step_020000.pt'}
GP_CFG = ControlledGPConfig(**json.loads(Path('configs/controlled_gp_prior.json').read_text()))
FIRST, LAST = int(sys.argv[1]), int(sys.argv[2])
N_Z, N_SEL, SEQ = 15, 5, 34
OUT = Path(f'tmp/logs/sim_groundtruth_classification_{PRIOR}'); OUT.mkdir(parents=True, exist_ok=True)


def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ---------------------------------------------------------------- tasks (deterministic in the task index)
@torch.no_grad()
def make_task(i):
    rng = np.random.default_rng([777, i])
    gen = torch.Generator(device=dev).manual_seed(777_000 + i)
    member = 'mlp' if PRIOR == 'tabpfn' else ('gp' if i % 2 == 0 else 'mlp')
    d = 1 if rng.random() < 0.5 else 2
    n = int(rng.integers(10, 31))
    if PRIOR == 'tabpfn':
        net = tabpfn.sample_networks(1, d, gen, dev)
        base = tabpfn.sample_inputs(net, SEQ, gen)                            # TabPFN's own IID input law
    else:
        z = _layout_points(1, SEQ, d, 0.75, gen, dev).float()
        if member == 'gp':
            base = z
        else:
            net = tabpfn.sample_networks(1, d, gen, dev)
            base = net['x_mean'][:, None] + net['x_std'][:, None] * z
    ctx = base[0, :n]
    mean, std = ctx.mean(0), ctx.std(0) + 1e-6
    dirs = torch.tensor(rng.normal(size=(2, d)), device=dev, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    far = mean + std * dirs * torch.tensor(rng.uniform(3, 4, (2, 1)), device=dev, dtype=torch.float32)
    x = torch.cat([base[0], far], 0)[None]                                     # 34 base + 2 far points
    if member == 'gp':
        logit, y, _ = sample_at_inputs(x.double(), GP_CFG, gen)
        p_true = torch.sigmoid(logit.double())[0]
    else:
        out = tabpfn.forward(net, x, gen)
        t = tabpfn.threshold(net, out[:, :SEQ], gen)
        y = (out > t[:, None]).float()
        p_true = tabpfn.label_probs(net, x, t, 64, gen)[0].double()
    x, y = x[0].double().cpu().numpy(), y[0].double().cpu().numpy()
    queries = [n, n + 1, SEQ, SEQ + 1]                                         # near, near, far, far
    zs = []
    for qi in queries:
        zq = []
        while len(zq) < N_Z:
            cand = x[qi] + rng.normal(0, 0.1 * std.cpu().numpy())
            if not any(np.allclose(cand, c) for c in zq):
                zq.append(cand)
        zs.append(np.array(zq))
    return {'i': i, 'member': member, 'd': d, 'n': n, 'x': x, 'y': y, 'p_true': p_true.cpu().numpy(),
            'queries': queries, 'zs': zs}


# ---------------------------------------------------------------- MDLM + VUD
def mdlm_p1(model, batch):
    """batch: list of (x rows (m, d), y rows with NaN at the target, target row) -> p(y=1)."""
    out = []
    for start in range(0, len(batch), 512):
        part = batch[start:start + 512]
        enc = []
        for x, y, row in part:
            xp = np.hstack([x, np.zeros((len(x), 2 - x.shape[1]))])          # 1-D tasks pad x2 = 0
            enc.append(model.tokenizer.encode(model.prepare_x(xp, [row]), y, [row], hide=True))
        tok = model.tokenizer
        ids, kinds, attention, values = tok.collate(enc, model.device)
        probs = answer_probs(model.model, tok, ids, kinds, attention, values, 1.0)
        for b, e in enumerate(enc):
            (pos,) = e['spans'][part[b][2]]
            pr = probs[b, pos, 12:12 + tok.labels].double().cpu().numpy()
            out.append(pr[1] / pr.sum())
    return np.array(out)


def vud(p1, pu, a, b):
    """VUD vanilla selection from p(y=1|x,D), p(u=1|z,D) and p(y=1|x,z,u=1/0,D) over the z's (bits)."""
    tu = H(p1)
    va = pu * H(a) + (1 - pu) * H(b)
    pxz = np.clip(pu * a + (1 - pu) * b, 1e-12, 1 - 1e-12)
    pc = np.clip(p1, 1e-12, 1 - 1e-12)
    kl = pc * np.log(pc / pxz) + (1 - pc) * np.log((1 - pc) / (1 - pxz))
    sel = np.argsort(kl, kind='stable')[:N_SEL]
    v = min(va[sel].min(), tu)
    return tu, v, tu - v


def mdlm_vud(model, task):
    x, y, n = task['x'], task['y'], task['n']
    cx, cy = x[:n], y[:n]
    rows = []
    for qi, zq in zip(task['queries'], task['zs']):
        xq = x[qi]
        batch = [(np.vstack([cx, xq]), np.append(cy, np.nan), n)]
        batch += [(np.vstack([cx, z]), np.append(cy, np.nan), n) for z in zq]
        for u in (1.0, 0.0):
            batch += [(np.vstack([cx, z, xq]), np.append(np.append(cy, u), np.nan), n + 1) for z in zq]
        p = mdlm_p1(model, batch)
        p1, pu, a, b = p[0], p[1:1 + N_Z], p[1 + N_Z:1 + 2 * N_Z], p[1 + 2 * N_Z:]
        rows.append((p1, *vud(p1, pu, a, b)))
    return rows


# ---------------------------------------------------------------- exact Bayes under the prior
def resample(w, keep, g):
    cdf = torch.cumsum(w, 0)
    return torch.searchsorted(cdf, torch.rand(keep, device=dev, generator=g) * cdf[-1]).clamp(max=len(w) - 1)


def gp_kernel(a, b, ls, same):
    k = torch.exp(-0.5 * ((a[:, :, None] - b[:, None]) ** 2).sum(-1) / ls[:, None, None] ** 2)
    if same:
        k = k + GP_CFG.jitter * torch.eye(a.shape[1], device=dev, dtype=a.dtype)
    return k / (1 + GP_CFG.jitter)


@torch.no_grad()
def gp_member(cz, cy, qz, g, hyper=60_000, funcs=4, chunk=15_000, keep=3000):
    n, d = cz.shape
    X = torch.tensor(cz, device=dev, dtype=torch.float64)
    sign = torch.tensor(2 * cy - 1, device=dev, dtype=torch.float64)
    parts = {k: [] for k in ('ls', 'sd', 't', 'f', 'logw')}
    for _ in range(hyper // chunk):
        hp = sample_hyperparameters(chunk, GP_CFG, g, dev)
        ls, sd, t = hp['lengthscale'], hp['noise_variance'].sqrt(), hp['threshold']
        Xe = X.expand(chunk, n, d)
        L = torch.linalg.cholesky(gp_kernel(Xe, Xe, ls, True))
        f = (L @ torch.randn(chunk, n, funcs, device=dev, generator=g, dtype=torch.float64)).transpose(1, 2)
        logw = torch.special.log_ndtr(sign * (f - t[:, None, None]) / sd[:, None, None]).sum(-1)
        rep = lambda v: v[:, None].expand(chunk, funcs).reshape(-1)
        for k, v in zip(parts, (rep(ls), rep(sd), rep(t), f.reshape(-1, n), logw.reshape(-1))):
            parts[k].append(v)
    ls, sd, t, f, logw = (torch.cat(parts[k]) for k in parts)
    w = torch.exp(logw - logw.max())
    idx = resample(w, keep, g)
    S = len(idx)
    Xe = X.expand(S, n, d)
    L = torch.linalg.cholesky(gp_kernel(Xe, Xe, ls[idx], True))
    alpha = torch.cholesky_solve(f[idx][..., None], L)
    Q = torch.tensor(qz, device=dev, dtype=torch.float64).expand(S, *qz.shape)
    kqc = gp_kernel(Q, Xe, ls[idx], False)
    v = torch.linalg.solve_triangular(L, kqc.transpose(1, 2), upper=False)
    cov = gp_kernel(Q, Q, ls[idx], True) - v.transpose(1, 2) @ v + 1e-9 * torch.eye(len(qz), device=dev, dtype=torch.float64)
    fq = (kqc @ alpha + torch.linalg.cholesky(cov) @ torch.randn(S, len(qz), 1, device=dev, generator=g, dtype=torch.float64)).squeeze(-1)
    p = torch.special.ndtr((fq - t[idx][:, None]) / sd[idx][:, None])
    return float(torch.logsumexp(logw, 0) - math.log(len(logw))), p.T.cpu().numpy(), float(w.sum() ** 2 / (w ** 2).sum())


def mlp_chunk(seed, cz, d, chunk=2000, draws=32):
    g = torch.Generator(device=dev).manual_seed(seed)
    net = tabpfn.sample_networks(chunk, d, g, dev)
    ctx = torch.tensor(cz, device=dev, dtype=torch.float32)
    xc = net['x_mean'][:, None] + net['x_std'][:, None] * ctx[None]
    out = tabpfn.forward(net, torch.cat([xc, tabpfn.sample_inputs(net, SEQ - len(cz), g)], 1), g)
    t = tabpfn.threshold(net, out, g)
    return net, t, tabpfn.label_probs(net, xc, t, draws, g).clamp(0.5 / draws, 1 - 0.5 / draws)


@torch.no_grad()
def mlp_member(cz, cy, qz, task_seed, chunks=10, keep=1000):
    d = cz.shape[1]
    yc = torch.tensor(cy, device=dev).bool()
    logw = []
    for c in range(chunks):
        _, _, p = mlp_chunk(task_seed * 1000 + c, cz, d)
        logw.append(torch.nan_to_num(torch.where(yc, p.log(), (1 - p).log()).sum(-1).double(), nan=-math.inf))
    logw = torch.cat(logw)
    w = torch.exp(logw - logw.max())
    idx = resample(w, keep, torch.Generator(device=dev).manual_seed(task_seed)).sort().values
    probs = []
    g = torch.Generator(device=dev).manual_seed(task_seed + 1)
    for c in idx.div(2000, rounding_mode='floor').unique().tolist():
        net, t, _ = mlp_chunk(task_seed * 1000 + c, cz, d)
        local = idx[(idx >= c * 2000) & (idx < (c + 1) * 2000)] - c * 2000
        sub = {k: v[local] for k, v in net.items()}
        q = torch.tensor(qz, device=dev, dtype=torch.float32)
        xq = (sub['x_mean'][:, None] + sub['x_std'][:, None] * q[None]).contiguous()
        probs.append(tabpfn.label_probs(sub, xq, t[local], 32, g).double())
    return float(torch.logsumexp(logw, 0) - math.log(len(logw))), torch.cat(probs).T.cpu().numpy(), float(w.sum() ** 2 / (w ** 2).sum())


def bayes_vud(task):
    """Exact-posterior TU, Bayes AU/EU and VUD on the exact posterior, per query (bits)."""
    x, y, n = task['x'], task['y'], task['n']
    pts = np.vstack([x[:n]] + [np.vstack([x[qi], zq]) for qi, zq in zip(task['queries'], task['zs'])])
    zall = normalize_by_context(pts, list(range(n)))
    cz, qz = zall[:n], zall[n:]
    g = torch.Generator(device=dev).manual_seed(task['i'])
    lz_mlp, p_mlp, ess_mlp = mlp_member(cz, y[:n], qz, 9_000 + task['i'], chunks=10)
    if PRIOR == 'tabpfn':                                                     # the generating prior is the MLP prior alone
        post, ess_gp, p, w = np.array([0.0, 1.0]), float('nan'), p_mlp, np.full(p_mlp.shape[1], 1 / p_mlp.shape[1])
    else:
        lz_gp, p_gp, ess_gp = gp_member(cz, y[:n], qz, g)
        lw = np.array([math.log(.5) + lz_gp, math.log(.5) + lz_mlp])
        post = np.exp(lw - lw.max()); post /= post.sum()
        p = np.concatenate([p_gp, p_mlp], 1)
        w = np.concatenate([np.full(p_gp.shape[1], post[0] / p_gp.shape[1]), np.full(p_mlp.shape[1], post[1] / p_mlp.shape[1])])
    rows = []
    for k in range(len(task['queries'])):
        block = p[k * (1 + N_Z):(k + 1) * (1 + N_Z)]
        px, pz = block[0], block[1:]
        p1 = px @ w
        au = H(px) @ w
        pu = pz @ w
        a = (pz * px) @ w / np.maximum(pu, 1e-12)
        b = ((1 - pz) * px) @ w / np.maximum(1 - pu, 1e-12)
        tu, va, ve = vud(p1, pu, a, b)
        rows.append((p1, tu, au, tu - au, va, ve, post[0], ess_gp, ess_mlp))
    return rows


models = {name: NumericMDLM(path, device=dev) for name, path in MODELS.items()}
rows = []
for i in range(FIRST, LAST):
    task = make_task(i)
    m_rows = {name: mdlm_vud(m, task) for name, m in models.items()}
    b_rows = bayes_vud(task)
    for k, qi in enumerate(task['queries']):
        rows.append({'task': i, 'member': task['member'], 'd': task['d'], 'n': task['n'], 'query': 'near' if k < 2 else 'far',
                     'y': task['y'][qi], 'p_true': task['p_true'][qi], 'true_AU': float(H(task['p_true'][qi])),
                     **{f'{name}_{key}': m_rows[name][k][j] for name in models for j, key in enumerate(('p1', 'TU', 'Va', 'Ve'))},
                     'bayes_p1': b_rows[k][0], 'bayes_TU': b_rows[k][1], 'bayes_AU': b_rows[k][2], 'bayes_EU': b_rows[k][3],
                     'ideal_Va': b_rows[k][4], 'ideal_Ve': b_rows[k][5], 'P_gp_member': b_rows[k][6],
                     'ess_gp': b_rows[k][7], 'ess_mlp': b_rows[k][8]})
    print(f'task {i} ({task["member"]}, d={task["d"]}, n={task["n"]}) done', flush=True)
    pd.DataFrame(rows).to_csv(OUT/f'queries_{FIRST}_{LAST}.csv', index=False)
