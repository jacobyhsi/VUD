"""Is the candidate over-confident far from the context on held-out tasks from its OWN prior?

Held-out test tasks (last 10,000 of the bank) of the candidate's geometry bank: context = the ordinary
observed rows (the auxiliary is left out), query = the episode's final query, whose true P(y=1) is
known (sigmoid of the stored latent; exact for GP tasks, a 64-pass estimate for MLP tasks). Queries are
binned by distance from the context (max |z-score| over features, after the model's own context
normalisation). Calibration against the prior's own tasks: a Bayes-optimal predictor's confidence
equals its expected accuracy on average, so confidence - accuracy > 0 means over-confident.
The same distances are reported for the moons_1 VUD grid, to show where moons sits.
Usage (from dllm-training/): python tmp/scripts/farfield_calibration.py [label=checkpoint ...]  (adds models)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from small_mdlm.backend import NumericMDLM
from small_mdlm.episodes import normalize_by_context

BANK = Path('data_banks/controlled_gp50_episode_pairs_1d_3m/geometry')
MODELS = {'candidate (Geometry1D 0.8M)': 'outputs/tabpfn_controlled_gp50_geometry_1d/checkpoint_step_018000.pt',
          'ControlledGP50 baseline': 'outputs/tabpfn_controlled_gp50_classify2d/checkpoint.pt'}
MODELS.update(dict(a.split('=', 1) for a in sys.argv[1:]))
BINS = [(0, 1.5), (1.5, 3), (3, 5), (5, np.inf)]
TASKS = 10_000
H = lambda p: -(np.clip(p, 1e-9, 1) * np.log2(np.clip(p, 1e-9, 1)) + np.clip(1 - p, 1e-9, 1) * np.log2(np.clip(1 - p, 1e-9, 1)))

arr = {k: np.load(BANK/f'{k}.npy', mmap_mode='r') for k in ('x', 'y', 'latent', 'context_size', 'query_count', 'dims', 'broad_query')}
total = len(arr['y'])
idx = np.arange(total - TASKS, total)
episodes = []
for i in idx:
    n, q, d = int(arr['context_size'][i]), int(arr['query_count'][i]), int(arr['dims'][i])
    ctx = n - (q == 1)                      # ordinary observed rows; the auxiliary is excluded
    query = n + q - 1
    rows = list(range(ctx)) + [query]
    x = np.asarray(arr['x'][i], float)[rows]
    y = np.asarray(arr['y'][i], float)[rows]
    p_true = 1 / (1 + np.exp(-float(arr['latent'][i][query])))
    z = normalize_by_context(x, list(range(ctx)))
    episodes.append((x, y, ctx, p_true, np.abs(z[-1, :d]).max(), d))
dist = np.array([e[4] for e in episodes])
print(f'{TASKS} held-out tasks; query distance from context (max |z|): '
      + ', '.join(f'{lo:g}-{hi:g}: {((dist >= lo) & (dist < hi)).mean():.0%}' for lo, hi in BINS))

# Where the moons_1 VUD grid's far-field points sit on the same distance scale.
R = Path('../results/toy_classification/moons_1/Dream-v0-Base-7B/va_vanilla')
cx = pd.read_csv(R/'D_test.csv')[['x1', 'x2']].to_numpy(float)
g1 = np.arange(-1.5, 2.5 + 1e-9, 0.2)
grid = np.stack(np.meshgrid(g1, g1), -1).reshape(-1, 2)
zg = normalize_by_context(np.vstack([cx, grid]), list(range(len(cx))))[len(cx):]
far = np.linalg.norm(grid[:, None] - cx[None], axis=-1).min(1) >= .35
dg = np.abs(zg).max(1)[far]
print('moons_1 far-field grid points on the same scale: '
      + ', '.join(f'{lo:g}-{hi:g}: {((dg >= lo) & (dg < hi)).mean():.0%}' for lo, hi in BINS))

for label, ckpt in MODELS.items():
    model = NumericMDLM(ckpt, device='cuda')
    pm = np.array([model.label_probs(x, y, ctx)[1] for x, y, ctx, *_ in episodes])
    pt = np.array([e[3] for e in episodes])
    conf = np.maximum(pm, 1 - pm)
    acc = np.where(pm >= 0.5, pt, 1 - pt)              # expected accuracy under the true task
    kl = pt * np.log2(np.clip(pt, 1e-9, 1) / np.clip(pm, 1e-9, 1)) + (1 - pt) * np.log2(np.clip(1 - pt, 1e-9, 1) / np.clip(1 - pm, 1e-9, 1))
    print(f'\n== {label}')
    print(f'{"distance":>9s} | {"n":>5s} | confidence | accuracy | over-confidence | model entropy | true label entropy | KL(true||model)')
    for lo, hi in BINS:
        s = (dist >= lo) & (dist < hi)
        if s.any():
            print(f'{lo:>4g}-{hi:<4g} | {s.sum():5d} | {conf[s].mean():10.3f} | {acc[s].mean():8.3f} | {conf[s].mean() - acc[s].mean():+15.3f} | '
                  f'{H(pm[s]).mean():13.3f} | {H(pt[s]).mean():18.3f} | {kl[s].mean():.3f}')
