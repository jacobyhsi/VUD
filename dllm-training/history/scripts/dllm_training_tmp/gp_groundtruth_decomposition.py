"""Ground-truth GP posterior uncertainty decomposition vs the TabPFN-trained MDLM + VUD, on 1-D GP datasets.

Data follow TabPFN v1's original GP path (tabpfn/priors/fast_gp.py + flexible_categorical.py): x ~ N(0, 1),
y = f(x) + eps with f ~ GP(0, outputscale * RBF(lengthscale)) and eps ~ N(0, noise), label = 1[y > y at a random
point] (binary rank threshold; the random class flip does not change entropies). Hyperparameters are real draws from
TabPFN's original prior (tabpfn_gp_hyper_hook.py). f is sampled jointly at the 64 data points and a plotting grid, so
the TRUE p(y=1 | x) = Phi((f(x) - t) / sqrt(noise)) is known on the grid. Context: 20 of the 64 points (both classes
in context and remainder, as TabPFN's compatibility check requires).
Ground-truth GP posterior: the generating GP with its TRUE hyperparameters and threshold (oracle), conditioned on the
context labels (elliptical slice sampling over f at the context points, exact Gaussian conditional on the grid):
  TU = H(E p), AU = E H(p), EU = TU - AU (bits).
MDLM: outputs/tabpfn_original_classify2d (trained on TabPFN-original data) + VUD vanilla as in the toy experiments
(15 z ~ N(x, (0.1 * context std)^2), Va = min over the 5 lowest-KL z, capped at TU).
Usage (from dllm-training/): python tmp/scripts/gp_groundtruth_decomposition.py <draw ids, comma-separated> <out.png>
SHOW_MDLM=0 draws the ground-truth GP decomposition only.
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '.')
from small_mdlm.backend import NumericMDLM
from small_mdlm.model import answer_probs

dev = 'cuda'
DRAWS = json.load(open('tmp/logs/tabpfn_original/gp_hyper_draws.json'))
IDS, OUT = [int(i) for i in sys.argv[1].split(',')], Path(sys.argv[2])
N_DATA, N_CTX, N_Z, N_SEL = 64, 20, 15, 5
GRID = np.linspace(-4, 4, 161)
CKPT = 'outputs/tabpfn_original_classify2d/checkpoint.pt'
SHOW_MDLM = os.environ.get('SHOW_MDLM', '1') == '1'
COL = {'truth': '#0b0b0b', 'gp': '#2a78d6', 'mdlm': '#eb6834', 'muted': '#52514e', 'band': '#f0efec'}


def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def rbf(a, b, h):
    return h['outputscale'] * torch.exp(-0.5 * (a[:, None] - b[None]) ** 2 / h['lengthscale'] ** 2)


# ---------------------------------------------------------------- one task in TabPFN's GP path, truth on the grid
def make_task(h, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    grid = torch.tensor(GRID, device=dev, dtype=torch.float64)
    for _ in range(1000):
        x = torch.randn(N_DATA, generator=g, device=dev, dtype=torch.float64)          # sampling = 'normal'
        pts = torch.cat([x, grid])
        K = rbf(pts, pts, h) + 1e-8 * h['outputscale'] * torch.eye(len(pts), device=dev, dtype=torch.float64)
        f = torch.linalg.cholesky(K) @ torch.randn(len(pts), generator=g, device=dev, dtype=torch.float64)
        y_noisy = f[:N_DATA] + math.sqrt(h['noise']) * torch.randn(N_DATA, generator=g, device=dev, dtype=torch.float64)
        t = y_noisy[torch.randint(0, N_DATA, (1,), generator=g, device=dev)]          # MulticlassRank, 2 classes
        lab = (y_noisy > t).double()
        if 0 < lab[:N_CTX].sum() < N_CTX and 0 < lab[N_CTX:].sum() < N_DATA - N_CTX:  # compatibility check
            p_true = torch.special.ndtr((f[N_DATA:] - t) / math.sqrt(h['noise']))
            return {'x': x[:N_CTX].cpu().numpy(), 'y': lab[:N_CTX].cpu().numpy(), 't': float(t),
                    'p_true': p_true.cpu().numpy()}
    raise RuntimeError('no compatible dataset')


# ---------------------------------------------------------------- oracle GP posterior (elliptical slice sampling)
@torch.no_grad()
def gp_posterior(task, h, seed, chains=512, iters=600, keep=200):
    g = torch.Generator(device=dev).manual_seed(seed)
    xc = torch.tensor(task['x'], device=dev, dtype=torch.float64)
    sign = torch.tensor(2 * task['y'] - 1, device=dev, dtype=torch.float64)
    sd, t = math.sqrt(h['noise']), task['t']
    Kc = rbf(xc, xc, h) + 1e-8 * h['outputscale'] * torch.eye(N_CTX, device=dev, dtype=torch.float64)
    L = torch.linalg.cholesky(Kc)
    loglik = lambda f: torch.special.log_ndtr(sign * (f - t) / sd).sum(-1)
    prior = lambda: torch.randn(chains, N_CTX, generator=g, device=dev, dtype=torch.float64) @ L.T
    f, kept = prior(), []
    for it in range(iters):
        nu, logy = prior(), loglik(f) + torch.log(torch.rand(chains, generator=g, device=dev, dtype=torch.float64))
        theta = torch.rand(chains, generator=g, device=dev, dtype=torch.float64) * 2 * math.pi
        lo, hi, done = theta - 2 * math.pi, theta.clone(), torch.zeros(chains, dtype=torch.bool, device=dev)
        new = f.clone()
        for _ in range(60):                                                 # shrink the bracket until accepted
            prop = f * torch.cos(theta)[:, None] + nu * torch.sin(theta)[:, None]
            ok = (loglik(prop) > logy) & ~done
            new[ok], done = prop[ok], done | ok
            if done.all():
                break
            lo = torch.where(theta < 0, theta, lo)
            hi = torch.where(theta >= 0, theta, hi)
            theta = lo + torch.rand(chains, generator=g, device=dev, dtype=torch.float64) * (hi - lo)
        f = new
        if it >= iters - keep and (iters - it) % 10 == 0:
            kept.append(f.clone())
    fc = torch.cat(kept)                                                    # (S, N_CTX) posterior samples
    grid = torch.tensor(GRID, device=dev, dtype=torch.float64)
    kgc = rbf(grid, xc, h)
    mean = kgc @ torch.cholesky_solve(fc.T, L)                              # (G, S)
    var = (h['outputscale'] - (torch.linalg.solve_triangular(L, kgc.T, upper=False) ** 2).sum(0)).clamp(min=0)
    fg = mean + var.sqrt()[:, None] * torch.randn(mean.shape, generator=g, device=dev, dtype=torch.float64)
    p = torch.special.ndtr((fg - t) / sd).cpu().numpy()                     # (G, S) p(y=1 | x, f)
    tu, au = H(p.mean(1)), H(p).mean(1)
    return p.mean(1), tu, au, tu - au


# ---------------------------------------------------------------- MDLM + VUD vanilla
model = NumericMDLM(CKPT, device=dev) if SHOW_MDLM else None
tok = model.tokenizer if SHOW_MDLM else None


@torch.no_grad()
def mdlm_p1(seqs):
    out = []
    for s in range(0, len(seqs), 1024):
        enc = []
        for x, y, row in seqs[s:s + 1024]:
            xp = np.stack([x, np.zeros_like(x)], 1)                         # 1-D: x2 = 0, as TabPFN zero-pads
            enc.append(tok.encode(model.prepare_x(xp, [row]), y, [row], hide=True))
        ids, kinds, attention, values = tok.collate(enc, model.device)
        probs = answer_probs(model.model, tok, ids, kinds, attention, values, 1.0)
        for b, e in enumerate(enc):
            (pos,) = e['spans'][seqs[s + b][2]]
            pr = probs[b, pos, 12:12 + tok.labels].double().cpu().numpy()
            out.append(pr[1] / pr.sum())
    return np.array(out)


def mdlm_vud(task, seed):
    rng = np.random.default_rng(seed)
    cx, cy, n = task['x'], task['y'], N_CTX
    sd = 0.1 * cx.std()
    seqs, zs = [], []
    for xq in GRID:
        z = xq + rng.normal(0, sd, N_Z)
        zs.append(z)
        seqs.append((np.append(cx, xq), np.append(cy, np.nan), n))
        seqs += [(np.append(cx, zi), np.append(cy, np.nan), n) for zi in z]
        for u in (1.0, 0.0):
            seqs += [(np.append(cx, [zi, xq]), np.append(cy, [u, np.nan]), n + 1) for zi in z]
    p = mdlm_p1(seqs).reshape(len(GRID), 1 + 3 * N_Z)
    p1, pu, a, b = p[:, 0], p[:, 1:1 + N_Z], p[:, 1 + N_Z:1 + 2 * N_Z], p[:, 1 + 2 * N_Z:]
    tu = H(p1)
    va = pu * H(a) + (1 - pu) * H(b)
    pxz = np.clip(pu * a + (1 - pu) * b, 1e-12, 1 - 1e-12)
    pc = np.clip(p1, 1e-12, 1 - 1e-12)[:, None]
    kl = pc * np.log(pc / pxz) + (1 - pc) * np.log((1 - pc) / (1 - pxz))
    sel = np.argsort(kl, axis=1, kind='stable')[:, :N_SEL]
    va = np.minimum(np.take_along_axis(va, sel, 1).min(1), tu)
    return p1, tu, va, tu - va


# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(len(IDS), 4, figsize=(17, 3.3 * len(IDS)), sharex=True, squeeze=False)
summary = []
for r, i in enumerate(IDS):
    h = DRAWS[i]
    task = make_task(h, 1000 + i)
    gp_p, gp_tu, gp_au, gp_eu = gp_posterior(task, h, 2000 + i)
    m_p, m_tu, m_va, m_ve = mdlm_vud(task, 3000 + i) if SHOW_MDLM else (np.full(len(GRID), np.nan),) * 4
    true_au = H(task['p_true'])
    lo, hi = task['x'].mean() - 2 * task['x'].std(), task['x'].mean() + 2 * task['x'].std()
    label = (f"draw #{i}: lengthscale {h['lengthscale']:.3g}, outputscale {h['outputscale']:.3g}, noise {h['noise']:.0e}")
    panels = [('p(y=1 | x, D)', [('true p(y=1|x)', task['p_true'], 'truth'), ('GP posterior', gp_p, 'gp'), ('MDLM', m_p, 'mdlm')]),
              ('TU (bits)', [('GP posterior', gp_tu, 'gp'), ('MDLM', m_tu, 'mdlm')]),
              ('AU (bits)' + ('   [orange: VUD Va]' if SHOW_MDLM else ''), [('true AU', true_au, 'truth'), ('GP posterior', gp_au, 'gp'), ('MDLM + VUD (Va)', m_va, 'mdlm')]),
              ('EU (bits)' + ('   [orange: VUD Ve = TU - Va]' if SHOW_MDLM else ''), [('GP posterior', gp_eu, 'gp'), ('MDLM + VUD (Ve)', m_ve, 'mdlm')])]
    if not SHOW_MDLM:
        panels = [(t, [ln for ln in lines if ln[2] != 'mdlm']) for t, lines in panels]
    for c, (title, lines) in enumerate(panels):
        ax = axes[r, c]
        ax.axvspan(lo, hi, color=COL['band'], zorder=0, lw=0)
        for name, v, k in lines:
            ax.plot(GRID, v, color=COL[k], lw=2 if k != 'truth' else 1.5, ls='--' if k == 'truth' else '-', label=name, zorder=3)
        if c == 0:
            ax.scatter(task['x'], task['y'], s=22, color=COL['muted'], zorder=4, label='context labels',
                       edgecolor='white', linewidth=1)
            ax.set_ylabel(f"draw #{i}\nlengthscale {h['lengthscale']:.2g}\noutputscale {h['outputscale']:.2g}\nnoise {h['noise']:.0e}",
                          color=COL['truth'], fontsize=9)
        ax.set_ylim(-0.04, 1.04)
        if r == 0:
            ax.set_title(title, fontsize=11, loc='left', color=COL['truth'])
        if r == len(IDS) - 1:
            ax.set_xlabel('x')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#c3c2b7')
        ax.tick_params(colors=COL['muted'], labelsize=8)
        ax.grid(axis='y', color='#e7e6e2', lw=0.6, zorder=0)
    inside = (GRID >= lo) & (GRID <= hi)
    for where, mask in (('inside', inside), ('outside', ~inside)):
        summary.append({'draw': i, 'region': where, 'true_AU': true_au[mask].mean(), 'gp_TU': gp_tu[mask].mean(),
                        'gp_AU': gp_au[mask].mean(), 'gp_EU': gp_eu[mask].mean(), 'mdlm_TU': m_tu[mask].mean(),
                        'mdlm_Va': m_va[mask].mean(), 'mdlm_Ve': m_ve[mask].mean()})
    print(f'draw {i} done', flush=True)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=COL['truth'], ls='--', lw=1.5), Line2D([], [], color=COL['gp'], lw=2),
           Line2D([], [], color=COL['mdlm'], lw=2),
           Line2D([], [], color=COL['muted'], marker='o', ls='', markersize=5, markeredgecolor='white')]
labels = ['truth (true p, true AU)', 'ground-truth GP posterior (true hyperparameters)',
          'MDLM + VUD (MDLM trained on TabPFN data; AU = VUD Va, EU = VUD Ve)', 'context labels']
if not SHOW_MDLM:
    handles, labels = [handles[0], handles[1], handles[3]], [labels[0], labels[1], labels[3]]
fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, -0.005, 'Grey band: context mean ± 2 std. GP posterior = the generating GP with its true hyperparameters and '
         'threshold, conditioned on the 20 context labels.', ha='center', fontsize=8, color=COL['muted'])
fig.tight_layout(rect=(0, 0.01, 1, 0.97))
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
import pandas as pd
pd.DataFrame(summary).round(3).to_csv(OUT.with_suffix('.csv'), index=False)
print(pd.DataFrame(summary).round(3).to_string(index=False))
