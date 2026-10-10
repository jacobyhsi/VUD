"""Exact-posterior reference for VUD's conditional predictions under the ControlledGP50 prior.

Prior (bank tabpfn_controlled_gp50_cls_3m): 50% controlled GP (small_mdlm/vendor/controlled_gp_prior.py,
configs/controlled_gp_prior.json) + 50% TabPFN v1 MLP, non-causal (small_mdlm/vendor/tabpfn_prior.py).
At every grid x and each of its 15 auxiliary z (shared by all MDLM VUD runs), Monte Carlo posterior
draws give p(y|x,D), p(u|z,D), p(y|x,z,u,D), the response |p(y|x,z,u=1,D) - p(y|x,z,u=0,D)| and
Va(z); compared with the MDLM runs' own values at the same (x, z), at all z and at the z each run's
VUD rule selected (Va minimum among the 5 lowest-KL z, ties included, uncapped in 2-D).

Approximations (reported, not hidden): inputs are the context z-scores, used directly as the GP's raw
N(0,1) coordinates and mapped to each MLP's input law as mean + std * z; the MLP's hidden noise is
integrated with 32/64 passes (floored at half a pass); Monte Carlo ESS is printed per member. The
controlled GP's threshold is independent of the data, so its likelihood is exact (probit).
Usage (from dllm-training/): python tmp/scripts/conditional_reference.py <logistic_regression|moons_1>
  [GP hyper draws, default 1,000,000] [MLP chunks of 2,000 networks, default 500]
Environment: SCORE_ONLY=1 reuses the saved reference.csv (no sampling) and only scores runs;
EXTRA_MODELS (comma list of results names) adds runs to the comparison; CLEAN_PROB overrides the controlled GP's
clean_probability (a prior variant; outputs then go to conditional_reference_<dataset>_clean<p>/).
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
from evaluate_vud_classification import true_p1
from small_mdlm.episodes import normalize_by_context
from small_mdlm.vendor import tabpfn_prior as tabpfn
from small_mdlm.vendor.controlled_gp_prior import ControlledGPConfig, sample_hyperparameters

torch.set_default_dtype(torch.float64)
dev = 'cuda'
DATASET = sys.argv[1]
GP_HYPER = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
MLP_CHUNKS = int(sys.argv[3]) if len(sys.argv) > 3 else 500
R = Path('../results/toy_classification')/DATASET
MODELS = ['MDLM-TabPFN-ControlledGP50-0.8M', 'MDLM-TabPFN-GPMLP50-0.8M', 'MDLM-TabPFNClassify2D-0.8M',
          'MDLM-TabPFNClassify2D-Geometry-0.8M', 'MDLM-BayesClassify2D-0.8M', 'MDLM-BayesClassify2D-2.5M']
MODELS += [m for m in os.environ.get('EXTRA_MODELS', '').split(',') if m]
SCORE_ONLY = os.environ.get('SCORE_ONLY') == '1'
COLS = ['x1'] if DATASET == 'logistic_regression' else ['x1', 'x2']
D = len(COLS)
GP_SETTINGS = json.loads(Path('configs/controlled_gp_prior.json').read_text())
if os.environ.get('CLEAN_PROB'):
    GP_SETTINGS['clean_probability'] = float(os.environ['CLEAN_PROB'])
CFG = ControlledGPConfig(**GP_SETTINGS)
WEIGHTS = {'controlled_gp': 0.5, 'tabpfn': 0.5}
KEEP = {'controlled_gp': 20_000, 'tabpfn': 4_000}
OUT = Path(f'tmp/logs/conditional_reference_{DATASET}' + (f"_clean{os.environ['CLEAN_PROB']}" if os.environ.get('CLEAN_PROB') else ''))


def resample(w, keep, g):
    cdf = torch.cumsum(w, 0)
    return torch.searchsorted(cdf, torch.rand(keep, device=dev, generator=g) * cdf[-1]).clamp(max=len(w) - 1)


def gp_kernel(a, b, ls, same):
    """Controlled GP covariance: (RBF + jitter on the diagonal) / (1 + jitter), as in sample_at_inputs."""
    k = torch.exp(-0.5 * ((a[:, :, None] - b[:, None]) ** 2).sum(-1) / ls[:, None, None] ** 2)
    if same:
        k = k + CFG.jitter * torch.eye(a.shape[1], device=dev)
    return k / (1 + CFG.jitter)


@torch.no_grad()
def gp_member(cz, cy, funcs=50, chunk=20_000, seed=3):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(cy)
    ctx = torch.tensor(cz, device=dev)
    sign = torch.tensor(2 * cy - 1, device=dev)
    parts = {k: [] for k in ('ls', 'sd', 't', 'f', 'logw')}
    for _ in range(GP_HYPER // chunk):
        hp = sample_hyperparameters(chunk, CFG, g, dev)
        ls, sd, t = hp['lengthscale'], hp['noise_variance'].sqrt(), hp['threshold']
        L = torch.linalg.cholesky(gp_kernel(ctx.expand(chunk, n, D), ctx.expand(chunk, n, D), ls, True))
        f = (L @ torch.randn(chunk, n, funcs, device=dev, generator=g)).transpose(1, 2)    # (chunk, funcs, n)
        logw = torch.special.log_ndtr(sign * (f - t[:, None, None]) / sd[:, None, None]).sum(-1)
        rep = lambda v: v[:, None].expand(chunk, funcs).reshape(-1)
        for k, v in zip(parts, (rep(ls), rep(sd), rep(t), f.reshape(-1, n), logw.reshape(-1))):
            parts[k].append(v)
    ls, sd, t, f, logw = (torch.cat(parts[k]) for k in parts)
    w = torch.exp(logw - logw.max())
    idx = resample(w, KEEP['controlled_gp'], g)
    P = ctx.expand(len(idx), n, D)
    L = torch.linalg.cholesky(gp_kernel(P, P, ls[idx], True))
    return {'log_z': float(torch.logsumexp(logw, 0) - math.log(len(logw))), 'ess': float(w.sum() ** 2 / (w ** 2).sum()),
            'draws': len(logw), 'ls': ls[idx], 'sd': sd[idx], 't': t[idx], 'pts': P, 'L': L,
            'alpha': torch.cholesky_solve(f[idx][..., None], L).squeeze(-1)}


def mlp_chunk(seed, ctx, chunk=2_000, draws=32):
    g = torch.Generator(device=dev).manual_seed(seed)
    net = tabpfn.sample_networks(chunk, D, g, dev)
    xc = net['x_mean'][:, None] + net['x_std'][:, None] * ctx[None]
    out = tabpfn.forward(net, torch.cat([xc, tabpfn.sample_inputs(net, 34 - len(ctx), g)], 1), g)
    t = tabpfn.threshold(net, out, g)
    return net, t, tabpfn.label_probs(net, xc, t, draws, g).clamp(0.5 / draws, 1 - 0.5 / draws)


@torch.no_grad()
def mlp_member(cz, cy, seed=1, chunk=2_000):
    torch.set_default_dtype(torch.float32)
    ctx = torch.tensor(cz, device=dev, dtype=torch.float32)
    yc = torch.tensor(cy, device=dev).bool()
    logw = []
    for c in range(MLP_CHUNKS):
        _, _, p = mlp_chunk(seed * 100_003 + c, ctx)
        logw.append(torch.nan_to_num(torch.where(yc, p.log(), (1 - p).log()).sum(-1).double(), nan=-math.inf))
    logw = torch.cat(logw)
    w = torch.exp(logw - logw.max())
    idx = resample(w, KEEP['tabpfn'], torch.Generator(device=dev).manual_seed(seed)).sort().values
    nets, ts = [], []
    for c in idx.div(chunk, rounding_mode='floor').unique().tolist():
        net, t, _ = mlp_chunk(seed * 100_003 + c, ctx)
        local = idx[(idx >= c * chunk) & (idx < (c + 1) * chunk)] - c * chunk
        nets.append({k: v[local] for k, v in net.items()}); ts.append(t[local])
    torch.set_default_dtype(torch.float64)
    return {'log_z': float(torch.logsumexp(logw, 0) - math.log(len(logw))), 'ess': float(w.sum() ** 2 / (w ** 2).sum()),
            'draws': len(logw), 'net': {k: torch.cat([n[k] for n in nets]) for k in nets[0]}, 't': torch.cat(ts)}


@torch.no_grad()
def member_probs(name, m, qz, g):
    """Per-draw p(y=1) at the query points (x first, then its z's): (Q, draws)."""
    if name == 'tabpfn':
        q = torch.tensor(qz, device=dev, dtype=torch.float32)
        q = (m['net']['x_mean'][:, None] + m['net']['x_std'][:, None] * q[None]).contiguous()
        return tabpfn.label_probs(m['net'], q, m['t'], 64, g).double().T
    S = len(m['t'])
    q = torch.tensor(qz, device=dev).expand(S, *qz.shape)
    kqc = gp_kernel(q, m['pts'], m['ls'], False)
    v = torch.linalg.solve_triangular(m['L'], kqc.transpose(1, 2), upper=False)
    cov = gp_kernel(q, q, m['ls'], True) - v.transpose(1, 2) @ v + 1e-10 * torch.eye(len(qz), device=dev)
    fq = (kqc @ m['alpha'][..., None] + torch.linalg.cholesky(cov) @ torch.randn(S, len(qz), 1, device=dev, generator=g)).squeeze(-1)
    return torch.special.ndtr((fq - m['t'][:, None]) / m['sd'][:, None]).T


def bits(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def selected_rows(group):
    """The production VUD choice: Va minimum among the 5 lowest-KL z (ties included)."""
    cand = group.loc[group.kl_pyx_pyxz.isin(group.kl_pyx_pyxz.nsmallest(5))]
    return cand.Va.idxmin()


ctx = pd.read_csv(R/MODELS[0]/'va_vanilla/D_test.csv')
cx, cy = ctx[COLS].to_numpy(float), ctx['label'].to_numpy(float)
zscore = lambda pts: normalize_by_context(np.vstack([cx, pts]), list(range(len(cx))))[len(cx):]
cz = normalize_by_context(cx, list(range(len(cx))))
base = pd.read_csv(R/MODELS[0]/'va_vanilla/results_test.csv')
if SCORE_ONLY:
    ref = pd.read_csv(OUT/'reference.csv')
    report = json.loads((OUT/'summary.json').read_text())['reference']
else:
    ref, report = None, None
members = {} if SCORE_ONLY else {'controlled_gp': gp_member(cz, cy), 'tabpfn': mlp_member(cz, cy)}
logs = {k: math.log(WEIGHTS[k]) + m['log_z'] for k, m in members.items()} or {'none': 0.0}
top = max(logs.values())
post = {k: math.exp(v - top) for k, v in logs.items()}
post = {k: v / sum(post.values()) for k, v in post.items()}
report = report or {'dataset': DATASET, 'members': {k: {'log10_evidence': m['log_z'] / math.log(10), 'ess': m['ess'], 'draws': m['draws'],
                                              'posterior_weight': post[k]} for k, m in members.items()}}
print(json.dumps(report['members'], indent=1), flush=True)

rows = []
for i, (key, grp) in enumerate([] if SCORE_ONLY else base.groupby([f'x_{c}' for c in COLS], sort=False)):
    xq = np.atleast_1d(key).astype(float)
    qz = zscore(np.vstack([xq, grp[[f'z_{c}' for c in COLS]].to_numpy(float)]))
    g = torch.Generator(device=dev).manual_seed(i)
    probs = [member_probs(k, m, qz, g) for k, m in members.items()]
    w = torch.cat([torch.full((p.shape[1],), post[k] / p.shape[1], device=dev) for k, p in zip(members, probs)])
    p = torch.cat(probs, 1)
    px, pz = p[0], p[1:]
    p1 = float(px @ w)
    pu = (pz @ w).cpu().numpy()
    a = ((pz * px) @ w).cpu().numpy() / np.maximum(pu, 1e-12)
    b = (((1 - pz) * px) @ w).cpu().numpy() / np.maximum(1 - pu, 1e-12)
    va = pu * bits(a) + (1 - pu) * bits(b)
    for j, (_, r) in enumerate(grp.iterrows()):
        rows.append({'row': r.name, **{c: r[f'x_{c}'] for c in COLS}, **{f'z_{c}': r[f'z_{c}'] for c in COLS}, 'z_index': j,
                     'bayes_p1': p1, 'bayes_TU': float(bits(np.array(p1))), 'bayes_pu': pu[j],
                     'bayes_p_u1': a[j], 'bayes_p_u0': b[j], 'bayes_response': abs(a[j] - b[j]), 'bayes_Va': va[j]})
if not SCORE_ONLY:
    ref = pd.DataFrame(rows).sort_values('row').reset_index(drop=True)   # same row order as every run's CSV
    OUT.mkdir(parents=True, exist_ok=True)
    ref.to_csv(OUT/'reference.csv', index=False)
assert (ref.row.to_numpy() == np.arange(len(base))).all()

x = ref.groupby(COLS, sort=False).first().reset_index()[COLS].to_numpy()
near_of = lambda pts: (((pts[:, 0] >= cx[:, 0].min()) & (pts[:, 0] <= cx[:, 0].max())) if D == 1
                       else np.linalg.norm(pts[:, None] - cx[None], axis=-1).min(1) < .35)
summary = {'reference': report}
ideal = ref.groupby(COLS, sort=False).agg(TU=('bayes_TU', 'first'), Va=('bayes_Va', 'min')).reset_index()
ideal_near = near_of(ideal[COLS].to_numpy())
summary['bayes'] = {r: {'response_all_z': float(ref.bayes_response[near_of(ref[COLS].to_numpy()) == (r == 'near')].mean()),
                        'Ve_min_Va': float((ideal.TU - ideal.Va)[ideal_near == (r == 'near')].mean())} for r in ('near', 'away')}
for model in MODELS:
    path = R/model/'va_vanilla/results_test.csv'
    if not path.exists():
        continue
    run = pd.read_csv(path)
    assert np.allclose(run[[f'z_{c}' for c in COLS]].to_numpy(), ref[[f'z_{c}' for c in COLS]].to_numpy()), model
    resp = (run['p(y=1|x,u=1,z,D)'] - run['p(y=1|x,u=0,z,D)']).abs().to_numpy()
    near_z = near_of(run[[f'x_{c}' for c in COLS]].to_numpy())
    sel = run.groupby([f'x_{c}' for c in COLS], sort=False).apply(selected_rows, include_groups=False).to_numpy()
    s = run.loc[sel]
    near_s = near_of(s[[f'x_{c}' for c in COLS]].to_numpy())
    va = s.Va.to_numpy() if D == 2 else np.minimum(s.Va.to_numpy(), s['H[p(y|x,D)]'].to_numpy())
    summary[model] = {}
    for region, mz, ms in (('near', near_z, near_s), ('away', ~near_z, ~near_s)):
        summary[model][region] = {
            'response_all_z': float(resp[mz].mean()), 'bayes_response_all_z': float(ref.bayes_response.to_numpy()[mz].mean()),
            'response_gap_all_z': float((resp - ref.bayes_response.to_numpy())[mz].mean()),
            'p1_abs_diff': float(np.abs(s['p(y=1|x,D)'].to_numpy() - ref.bayes_p1.to_numpy()[sel])[ms].mean()),
            'response_selected_z': float(resp[sel][ms].mean()), 'bayes_response_selected_z': float(ref.bayes_response.to_numpy()[sel][ms].mean()),
            'Ve_selected': float((s['H[p(y|x,D)]'].to_numpy() - va)[ms].mean()),
            'bayes_Ve_at_selected_z': float((ref.bayes_TU.to_numpy() - ref.bayes_Va.to_numpy())[sel][ms].mean())}
(OUT/('summary.json' if not SCORE_ONLY else 'summary_scored.json')).write_text(json.dumps(summary, indent=2) + '\n')
print(f'\n{DATASET}: Bayes reference Ve (min-Va z) near {summary["bayes"]["near"]["Ve_min_Va"]:.3f}, away {summary["bayes"]["away"]["Ve_min_Va"]:.3f}')
print(f'{"model":38s} region | response (model vs Bayes, all z) | at selected z | Ve selected (model vs Bayes) | |p1-Bayes|')
for model in MODELS:
    for region in ('near', 'away'):
        if model in summary:
            m = summary[model][region]
            print(f'{model:38s} {region:6s} | {m["response_all_z"]:.3f} vs {m["bayes_response_all_z"]:.3f} | '
                  f'{m["response_selected_z"]:.3f} vs {m["bayes_response_selected_z"]:.3f} | '
                  f'{m["Ve_selected"]:.3f} vs {m["bayes_Ve_at_selected_z"]:.3f} | {m["p1_abs_diff"]:.3f}')
print(f'Saved {OUT}/reference.csv and summary.json')
