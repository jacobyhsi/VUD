"""Exact-Bayes + ideal-VUD check of PFN prior bags on a VUD dataset, before training on them.

Members (all with 34-point datasets, as in the banks):
  gp_bernoulli  f ~ GP Matern-2.5 (ARD lengthscale ~ Gamma(3, 6), outputscale ~ Gamma(2, 0.15)) + noise 1e-3,
                x ~ U[0,1]^d, y ~ Bernoulli(sigmoid(f))
  gp_median     same f, y = 1[f > median of f over the dataset]
  bnn           small_mdlm/vendor/pfn_mlp_prior.py (paper's tabular BNN prior), x ~ N(0,1)^d,
                y = 1[out > median of out over the dataset]
  tabpfn        small_mdlm/vendor/tabpfn_prior.py (TabPFN v1 released MLP prior, non-causal branch),
                y = 1[out > rank or value threshold]
  tabpfn_gp     TabPFN v1's GP member (tabpfn/priors/fast_gp.py with the released checkpoint's
                differentiable hyperparameters): x ~ N(0,1)^d, f ~ GP(0, outputscale * RBF(lengthscale)),
                outputscale and lengthscale ~ meta_trunc_norm_log_scaled(1e-5, 10), observation noise
                variance in {1e-5, 1e-4, 1e-2} (softmax([1, w1, w2]), w ~ U(-3, 5)), y = 1[f + noise > rank
                or value threshold]; the observation noise is integrated exactly (probit likelihood)
The context is z-scored (as the MDLM sees it): GP members map it to their box as 0.5 + z / sqrt(12),
the BNN uses z directly and TabPFN maps it to each network's input law (mean + std * z). The dataset's 34 - n other points are drawn from each member's input law, and
the median is taken over the context and those points (not the query).
Each member's evidence Z = p(context labels | member) comes from prior draws: GP-Bernoulli and BNN
weighted by their likelihoods (MLP members' hidden noise integrated with 64 passes for the BNN and 32
for TabPFN, probabilities floored at half a pass), GP-median by acceptance. A bag's posterior over members is prior weight x Z. Each member is
resampled to `keep` draws; at x and its 15 z's (from the mixed-label MDLM run) this gives the exact
p(y|x,D), the Bayes split and VUD's rule on the exact posterior (bits), for several bags.
Usage (from dllm-training/): python tmp/scripts/mode1_bag.py <logistic_regression|moons_1> [TabPFN chunks, default 500]
Environment: MODE1_MEMBERS (comma list, default all members) and MODE1_TAG (suffix of the CSV/PNG names).
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
from bayes_gp_classification import entropy_bits
from evaluate_vud_classification import true_p1
from small_mdlm.episodes import normalize_by_context
from small_mdlm.vendor import pfn_mlp_prior as bnn
from small_mdlm.vendor import tabpfn_prior as tabpfn

torch.set_default_dtype(torch.float64)
dev = 'cuda'
SEQ, NOISE, OS, KEEP = 34, 1e-3, 2.0, 20_000
DATASET = sys.argv[1]
R = f'../results/toy_classification/{DATASET}/MDLM-PFNClassify2D-MixedLabels-0.8M/va_vanilla'
COLS = ['x1'] if DATASET == 'logistic_regression' else ['x1', 'x2']
D = len(COLS)
BAGS = {'GP Bernoulli + GP median (trained mixed prior)': {'gp_bernoulli': 0.5, 'gp_median': 0.5},
        'BNN alone': {'bnn': 1.0},
        'GP Bernoulli + GP median + BNN (uniform bag)': {'gp_bernoulli': 1 / 3, 'gp_median': 1 / 3, 'bnn': 1 / 3},
        'TabPFN alone': {'tabpfn': 1.0},
        'GP Bernoulli + TabPFN': {'gp_bernoulli': 0.5, 'tabpfn': 0.5},
        'GP Bernoulli + GP median + BNN + TabPFN (uniform bag)': {'gp_bernoulli': 0.25, 'gp_median': 0.25, 'bnn': 0.25, 'tabpfn': 0.25},
        'TabPFN GP alone': {'tabpfn_gp': 1.0},
        'TabPFN GP + TabPFN MLP': {'tabpfn_gp': 0.5, 'tabpfn': 0.5}}
WANTED = os.environ.get('MODE1_MEMBERS', 'gp_bernoulli,gp_median,bnn,tabpfn,tabpfn_gp').split(',')
BAGS = {k: v for k, v in BAGS.items() if set(v) <= set(WANTED)}
TAG = os.environ.get('MODE1_TAG', '')
bits = lambda p: -(np.clip(p, 1e-12, 1) * np.log2(np.clip(p, 1e-12, 1))
                   + np.clip(1 - p, 1e-12, 1) * np.log2(np.clip(1 - p, 1e-12, 1)))


def matern(a, b, ls, os):
    """a (S,P,d), b (S,Q,d), ls (S,d), os (S,) -> (S,P,Q)."""
    s = math.sqrt(5) * ((a[:, :, None] - b[:, None]) / ls[:, None, None]).norm(dim=-1)
    return os[:, None, None] * (1 + s + s ** 2 / 3) * torch.exp(-s)


def resample(w, keep, g):
    """Inverse-CDF resampling (torch.multinomial caps the category count at 2^24)."""
    cdf = torch.cumsum(w, 0)
    return torch.searchsorted(cdf, torch.rand(keep, device=dev, generator=g) * cdf[-1]).clamp(max=len(w) - 1)


@torch.no_grad()
def gp_members(cx_raw, cy, hyper=200_000, funcs=50, chunk=10_000, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(cy)
    ctx = torch.tensor(cx_raw, device=dev)
    yc = torch.tensor(cy, device=dev).bool()
    parts = {k: [] for k in ('ls', 'os', 'pts', 'f', 'logw', 'ok')}
    for _ in range(hyper // chunk):
        ls = torch._standard_gamma(torch.full((chunk, D), 3.0, device=dev), generator=g) / 6.0
        os = torch._standard_gamma(torch.full((chunk,), OS, device=dev), generator=g) / 0.15
        pts = torch.cat([ctx.expand(chunk, n, D), torch.rand(chunk, SEQ - n, D, device=dev, generator=g)], 1)
        K = matern(pts, pts, ls, os) + (NOISE + 1e-8) * torch.eye(SEQ, device=dev)
        f = (torch.linalg.cholesky(K) @ torch.randn(chunk, SEQ, funcs, device=dev, generator=g)).transpose(1, 2)
        fc = f[..., :n]
        logw = torch.where(yc, torch.nn.functional.logsigmoid(fc), torch.nn.functional.logsigmoid(-fc)).sum(-1)
        ok = ((fc > bnn.median_threshold(f)[..., None]) == yc).all(-1)
        for k, v in zip(parts, (ls, os, pts, f, logw.reshape(-1), ok.reshape(-1))):
            parts[k].append(v)
    ls, os, pts, f, logw, ok = (torch.cat(parts[k]) for k in parts)
    f = f.reshape(-1, SEQ)
    total = len(logw)
    w = torch.exp(logw - logw.max())
    members = {'gp_bernoulli': {'log_z': float(torch.logsumexp(logw, 0) - math.log(total)),
                                'idx': resample(w, KEEP, g), 'ess': float(w.sum() ** 2 / (w ** 2).sum())}}
    accepted = int(ok.sum())
    # No acceptance: report the 95% upper bound (3 / draws) on Z and drop the member.
    members['gp_median'] = {'log_z': math.log(max(accepted, 0) / total) if accepted else -math.inf,
                            'log_z_upper': math.log(max(accepted, 3) / total), 'accepted': accepted}
    if accepted:
        idx = ok.nonzero().squeeze(1)
        members['gp_median']['idx'] = idx[torch.randint(len(idx), (KEEP,), device=dev, generator=g)]
    for name, m in members.items():
        if 'idx' not in m:
            continue
        h = m['idx'] // funcs
        P, fs = pts[h], f[m['idx']]
        L = torch.linalg.cholesky(matern(P, P, ls[h], os[h]) + (NOISE + 1e-8) * torch.eye(SEQ, device=dev))
        m.update(ls=ls[h], os=os[h], pts=P, L=L, alpha=torch.cholesky_solve(fs[..., None], L).squeeze(-1),
                 t=bnn.median_threshold(fs))
        del m['idx']
    return members


def rbf(a, b, ls, os):
    """gpytorch ScaleKernel(RBFKernel): os * exp(-|a - b|^2 / (2 ls^2)); ls, os (S,)."""
    return os[:, None, None] * torch.exp(-0.5 * ((a[:, :, None] - b[:, None]) ** 2).sum(-1) / ls[:, None, None] ** 2)


@torch.no_grad()
def tabpfn_gp_member(cz, cy, hyper=200_000, funcs=50, chunk=10_000, seed=2):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(cy)
    ctx = torch.tensor(cz, device=dev)
    sign = torch.tensor(2 * cy - 1, device=dev)
    parts = {k: [] for k in ('ls', 'os', 'sd', 'pts', 'f', 't', 'logw')}
    for _ in range(hyper // chunk):
        ls = tabpfn._trunc_norm_log_scaled(chunk, 1e-5, 10.0, g, dev).clamp(min=1e-6)
        os_ = tabpfn._trunc_norm_log_scaled(chunk, 1e-5, 10.0, g, dev).clamp(min=1e-8)
        logits = torch.cat([torch.ones(chunk, 1, device=dev), -3 + 8 * torch.rand(chunk, 2, device=dev, generator=g)], 1)
        noise = torch.tensor([1e-5, 1e-4, 1e-2], device=dev)[torch.multinomial(torch.softmax(logits, 1), 1, generator=g).squeeze(1)]
        sd = noise.sqrt()
        pts = torch.cat([ctx.expand(chunk, n, D), torch.randn(chunk, SEQ - n, D, device=dev, generator=g)], 1)
        K = rbf(pts, pts, ls, os_) + (1e-6 * os_)[:, None, None] * torch.eye(SEQ, device=dev)
        f = (torch.linalg.cholesky(K) @ torch.randn(chunk, SEQ, funcs, device=dev, generator=g)).transpose(1, 2)
        y = f + sd[:, None, None] * torch.randn(f.shape, device=dev, generator=g)
        # Two-class threshold on the dataset's observed values: 'rank' (a random point's value) or 'value'
        # (normalized y > c, c ~ N(0, 1)), equally likely; a fair coin per dataset.
        pick = torch.randint(SEQ, (chunk, funcs, 1), device=dev, generator=g)
        rank_t = y.gather(2, pick).squeeze(2)
        value_t = y.mean(2) + y.std(2) * torch.randn(chunk, funcs, device=dev, generator=g)
        t = torch.where(torch.rand(chunk, funcs, device=dev, generator=g) < 0.5, rank_t, value_t)
        logw = torch.special.log_ndtr(sign * (f[..., :n] - t[..., None]) / sd[:, None, None]).sum(-1)
        for k, v in zip(parts, (ls, os_, sd, pts, f, t.reshape(-1), logw.reshape(-1))):
            parts[k].append(v)
    ls, os_, sd, pts, f, t, logw = (torch.cat(parts[k]) for k in parts)
    f = f.reshape(-1, SEQ)
    w = torch.exp(logw - logw.max())
    idx = resample(w, KEEP, g)
    h = idx // funcs
    P = pts[h]
    L = torch.linalg.cholesky(rbf(P, P, ls[h], os_[h]) + (1e-6 * os_[h])[:, None, None] * torch.eye(SEQ, device=dev))
    return {'tabpfn_gp': {'log_z': float(torch.logsumexp(logw, 0) - math.log(len(logw))),
                          'ess': float(w.sum() ** 2 / (w ** 2).sum()), 'ls': ls[h], 'os': os_[h], 'sd': sd[h],
                          'pts': P, 'L': L, 'alpha': torch.cholesky_solve(f[idx][..., None], L).squeeze(-1), 't': t[idx]}}


MLP = {'bnn': dict(module=bnn, draws=64, chunks=100, chunk=20_000, keep=KEEP),
       'tabpfn': dict(module=tabpfn, draws=32, chunks=int(sys.argv[2]) if len(sys.argv) > 2 else 500, chunk=2_000, keep=4_000)}   # TabPFN nets are ~2 MB each


def mlp_inputs(name, net, z):
    """Context/query z-scores in the network's input scale."""
    if name == 'bnn':
        return z.expand(len(net['noise']), *z.shape)
    return net['x_mean'][:, None] + net['x_std'][:, None] * z[None]


def mlp_chunk(name, seed, ctx):
    """Networks, threshold and the context's label probabilities for one chunk (deterministic in seed)."""
    spec = MLP[name]; mod, chunk, draws = spec['module'], spec['chunk'], spec['draws']
    g = torch.Generator(device=dev).manual_seed(seed)
    net = mod.sample_networks(chunk, D, g, dev)
    xc = mlp_inputs(name, net, ctx)
    other = (torch.randn(chunk, SEQ - len(ctx), D, device=dev, generator=g) if name == 'bnn'
             else mod.sample_inputs(net, SEQ - len(ctx), g))
    out = mod.forward(net, torch.cat([xc, other], 1), g)
    t = bnn.median_threshold(out) if name == 'bnn' else mod.threshold(net, out, g)
    p = mod.label_probs(net, xc, t, draws, g)
    return net, t, p.clamp(0.5 / draws, 1 - 0.5 / draws)


@torch.no_grad()
def mlp_member(name, cz, cy, seed=1):
    spec = MLP[name]; chunk = spec['chunk']
    torch.set_default_dtype(torch.float32)
    ctx = torch.tensor(cz, device=dev, dtype=torch.float32)
    yc = torch.tensor(cy, device=dev).bool()
    logw = []
    for c in range(spec['chunks']):
        _, _, p = mlp_chunk(name, seed * 100_003 + c, ctx)
        lw = torch.where(yc, p.log(), (1 - p).log()).sum(-1).double()
        logw.append(torch.nan_to_num(lw, nan=-math.inf))
    logw = torch.cat(logw)
    w = torch.exp(logw - logw.max())
    idx = resample(w, spec['keep'], torch.Generator(device=dev).manual_seed(seed)).sort().values
    nets, ts = [], []
    for c in idx.div(chunk, rounding_mode='floor').unique().tolist():   # regenerate only the chunks that were picked
        net, t, _ = mlp_chunk(name, seed * 100_003 + c, ctx)
        local = idx[(idx >= c * chunk) & (idx < (c + 1) * chunk)] - c * chunk
        nets.append({k: v[local] for k, v in net.items()}); ts.append(t[local])
    torch.set_default_dtype(torch.float64)
    return {name: {'log_z': float(torch.logsumexp(logw, 0) - math.log(len(logw))),
                   'ess': float(w.sum() ** 2 / (w ** 2).sum()),
                   'net': {k: torch.cat([n[k] for n in nets]) for k in nets[0]}, 't': torch.cat(ts)}}


@torch.no_grad()
def member_probs(name, m, q_raw, q_z, g):
    """Per-draw p(y=1) at the query points (x first, then its z's): (Q, KEEP)."""
    if name in MLP:
        q = mlp_inputs(name, m['net'], torch.tensor(q_z, device=dev, dtype=torch.float32)).contiguous()
        return MLP[name]['module'].label_probs(m['net'], q, m['t'], 64, g).double().T
    if name == 'tabpfn_gp':
        q = torch.tensor(q_z, device=dev).expand(KEEP, *q_z.shape)
        kqc = rbf(q, m['pts'], m['ls'], m['os'])
        v = torch.linalg.solve_triangular(m['L'], kqc.transpose(1, 2), upper=False)
        cov = rbf(q, q, m['ls'], m['os']) - v.transpose(1, 2) @ v + (1e-6 * m['os'])[:, None, None] * torch.eye(len(q_z), device=dev)
        fq = (kqc @ m['alpha'][..., None] + torch.linalg.cholesky(cov) @ torch.randn(KEEP, len(q_z), 1, device=dev, generator=g)).squeeze(-1)
        return torch.special.ndtr((fq - m['t'][:, None]) / m['sd'][:, None]).T
    q = torch.tensor(q_raw, device=dev).expand(KEEP, *q_raw.shape)
    kqc = matern(q, m['pts'], m['ls'], m['os'])
    v = torch.linalg.solve_triangular(m['L'], kqc.transpose(1, 2), upper=False)
    cov = matern(q, q, m['ls'], m['os']) - v.transpose(1, 2) @ v + (NOISE + 1e-8) * torch.eye(len(q_raw), device=dev)
    fq = (kqc @ m['alpha'][..., None] + torch.linalg.cholesky(cov) @ torch.randn(KEEP, len(q_raw), 1, device=dev, generator=g)).squeeze(-1)
    return (torch.sigmoid(fq) if name == 'gp_bernoulli' else (fq > m['t'][:, None]).double()).T


def kl(p, q):
    p, q = np.clip(p, 1e-12, 1 - 1e-12), np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def decompose(p, w):
    """Bayes p1, total, aleatoric and VUD's Va on the weighted draws p (Q, S) (row 0 = x)."""
    H = lambda v: float(entropy_bits(torch.tensor(v)))
    px = p[0]; p1 = float(px @ w)
    tu, ale = H(p1), float(entropy_bits(px) @ w)
    vas, kls = [], []
    for j in range(1, len(p)):
        pz = p[j]; pu1 = float(pz @ w)
        a = float((pz * px) @ w) / max(pu1, 1e-12)
        b = float(((1 - pz) * px) @ w) / max(1 - pu1, 1e-12)
        vas.append(pu1 * H(a) + (1 - pu1) * H(b)); kls.append(kl(p1, pu1 * a + (1 - pu1) * b))
    return p1, tu, ale, min(min(np.array(vas)[np.argsort(kls, kind='stable')[:5]]), tu)


ctx = pd.read_csv(f'{R}/D_test.csv')
cx, cy = ctx[COLS].to_numpy(float), ctx['label'].to_numpy(float)
runs = pd.read_csv(f'{R}/results_test.csv')
zscore = lambda points: normalize_by_context(np.vstack([cx, points]), list(range(len(cx))))
cz = zscore(np.empty((0, D)))
members = {}
if {'gp_bernoulli', 'gp_median'} & set(WANTED):
    members.update(gp_members(0.5 + cz / math.sqrt(12), cy))
for name in ('bnn', 'tabpfn'):
    if name in WANTED:
        members.update(mlp_member(name, cz, cy))
if 'tabpfn_gp' in WANTED:
    members.update(tabpfn_gp_member(cz, cy))
members = {k: v for k, v in members.items() if k in WANTED}
for name, m in members.items():
    print(f'{DATASET} {name:13s} log10 Z {m["log_z"] / math.log(10):8.3f}'
          + (f'  (95% upper bound {m["log_z_upper"] / math.log(10):.3f}, accepted {m["accepted"]})' if 'accepted' in m else '')
          + (f'  ESS {m["ess"]:.0f}' if 'ess' in m else ''))
posteriors = {}
for bag, prior in BAGS.items():
    logs = {k: math.log(v) + members[k]['log_z'] for k, v in prior.items()}
    top = max(logs.values())
    norm = sum(math.exp(v - top) for v in logs.values())
    posteriors[bag] = {k: math.exp(v - top) / norm for k, v in logs.items()}
    print(f'P(member | D) under [{bag}]: ' + ', '.join(f'{k} {v:.3f}' for k, v in posteriors[bag].items()))
rows = []
for i, (key, grp) in enumerate(runs.groupby([f'x_{c}' for c in COLS])):
    xq = np.atleast_1d(key).astype(float)
    q_z = zscore(np.vstack([xq, grp[[f'z_{c}' for c in COLS]].to_numpy(float)]))[len(cx):]
    g = torch.Generator(device=dev).manual_seed(i)
    probs = {name: member_probs(name, m, 0.5 + q_z / math.sqrt(12), q_z, g)
             for name, m in members.items() if 'L' in m or 'net' in m}
    for bag, post in posteriors.items():
        live = [k for k, v in post.items() if v > 0 and k in probs]
        p = torch.cat([probs[k] for k in live], 1)
        w = torch.cat([torch.full((probs[k].shape[1],), post[k] / probs[k].shape[1], device=dev) for k in live])
        rows.append((bag, *xq, *decompose(p, w / w.sum())))
d = pd.DataFrame(rows, columns=['bag'] + COLS + ['p1', 'TU', 'ale', 'va'])
stem = f'tmp/logs/mode1_bag_{DATASET}' + (f'_tabpfn{sys.argv[2]}' if len(sys.argv) > 2 else '') + (f'_{TAG}' if TAG else '')
d.to_csv(f'{stem}.csv', index=False)
for bag, s in d.groupby('bag', sort=False):
    x = s[COLS].to_numpy()
    truth = bits(true_p1(DATASET, x))
    near = ((x[:, 0] >= cx.min()) & (x[:, 0] <= cx.max()) if DATASET == 'logistic_regression'
            else np.min(np.linalg.norm(x[:, None] - cx[None], axis=-1), 1) < 0.35)
    print(f'== {bag}')
    for label, mask in (('near data', near), ('away from data', ~near)):
        t, b = truth[mask], s[mask]
        print(f'   {label:15s} ale/epi: exact Bayes {b.ale.mean():.2f}/{(b.TU - b.ale).mean():.2f} | '
              f'ideal VUD {b.va.mean():.2f}/{(b.TU - b.va).mean():.2f} | true {t.mean():.2f} | '
              f'|err| Bayes {np.abs(b.ale - t).mean():.3f}, ideal VUD {np.abs(b.va - t).mean():.3f}')
    # Epistemic growth: ideal-VUD Ve by distance from the nearest context point (in input units).
    dist = np.min(np.linalg.norm(x[:, None] - cx[None], axis=-1), 1)
    edges = (0, 0.35, 1.0, 2.0, np.inf) if DATASET == 'moons_1' else (0, 1e-9, 2.0, 5.0, np.inf)
    cells = [f'{lo:g}-{hi:g}: {(s.TU - s.va)[(dist >= lo) & (dist < hi)].mean():.2f}' for lo, hi in zip(edges, edges[1:])
             if ((dist >= lo) & (dist < hi)).any()]
    print('   ideal-VUD Ve by distance to data: ' + ' | '.join(cells))
if D == 2:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    bags = list(d.bag.unique())
    fig, axes = plt.subplots(len(bags), 3, figsize=(13, 4 * len(bags)), constrained_layout=True, squeeze=False)
    for row, bag in zip(axes, bags):
        s = d[d.bag == bag]
        for ax, (title, v) in zip(row, (('exact Bayes total', s.TU), ('ideal VUD Va', s.va), ('ideal VUD Ve', s.TU - s.va))):
            piv = s.assign(v=v).pivot(index='x2', columns='x1', values='v')
            mesh = ax.pcolormesh(piv.columns, piv.index, piv.to_numpy(), vmin=0, vmax=1, cmap='viridis', shading='auto')
            ax.scatter(cx[:, 0], cx[:, 1], c=cy, cmap='coolwarm', edgecolors='w', s=18)
            ax.set_title(f'{bag}: {title}', fontsize=9)
        fig.colorbar(mesh, ax=row)
    fig.savefig(f'{stem}.png', dpi=70)
