"""Is the regression MDLM over-confident far from the context on held-out tasks from its OWN prior?

Held-out test tasks (last 2,000 of the regression bank): context = the ordinary observed rows (the auxiliary is
left out), query = the episode's final query. x and y are z-scored with the context, as in training. For each task,
32 samples of y at the query (batched). Binned by the query's |z| distance from the context:
  predicted std (sample std), the actual RMS error of the sample mean against the observed y, their ratio
  (> 1 = over-confident), the share of observed y inside the central 90% sample interval (0.90 = calibrated),
  and the true noise std, all in context-normalized y units; split by prior member (GP / MLP).
The same distance scale is reported for the linear_regression and gaps VUD grids.
Usage (from dllm-training/): python tmp/scripts/regression_calibration.py <checkpoint> [bank]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
from small_mdlm.backend import NumericMDLM
from small_mdlm.episodes import normalize_by_context, normalize_targets
from small_mdlm.model import sample

CKPT = sys.argv[1]
BANK = Path(sys.argv[2] if len(sys.argv) > 2 else 'data_banks/regression_hetgp50_episodes_3m')
TASKS, K, BATCH = 2000, 32, 2048
BINS = [(0, 1.5), (1.5, 3), (3, 5), (5, np.inf)]

arr = {k: np.load(BANK/f'{k}.npy', mmap_mode='r') for k in ('x', 'y', 'mean', 'std', 'context_size', 'query_count', 'gp_task')}
total = len(arr['y'])
model = NumericMDLM(CKPT, device='cuda')
tok = model.tokenizer
examples, info = [], []
for i in range(total - TASKS, total):
    n, q = int(arr['context_size'][i]), int(arr['query_count'][i])
    ctx = n - (q == 1)
    rows = list(range(ctx)) + [n + q - 1]
    x = np.asarray(arr['x'][i], float)[rows]
    y = np.asarray(arr['y'][i], float)[rows]
    obs = list(range(ctx))
    xz = normalize_by_context(x, obs)
    yz, ym, ys = normalize_targets(y, obs)
    examples.append(tok.encode(xz, np.append(yz[:-1], np.nan), [ctx], hide=True))
    info.append((abs(xz[-1, 0]), yz[-1], (float(arr['mean'][i][n + q - 1]) - ym) / ys, float(arr['std'][i][n + q - 1]) / ys,
                 int(arr['gp_task'][i]), ctx))
samples = np.zeros((TASKS, K))
flat = [(t, k) for t in range(TASKS) for k in range(K)]
for start in range(0, len(flat), BATCH):
    part = flat[start:start + BATCH]
    ids, kinds, attention, values = tok.collate([examples[t] for t, _ in part], model.device)
    gen = sample(model.model, tok, ids, kinds, attention, seed=start, temperature=1.0, values=values,
                 steps=model.config['training']['sample_steps'])
    for b, (t, k) in enumerate(part):
        (positions,) = examples[t]['spans'][info[t][5]],
        samples[t, k] = tok.decode_number(gen[b, positions].tolist())
dist, y_obs, true_mean, true_std, gp, _ = map(np.array, zip(*info))
m, s = samples.mean(1), samples.std(1)
lo, hi = np.percentile(samples, 5, axis=1), np.percentile(samples, 95, axis=1)
print(f'{TASKS} held-out tasks, {K} samples each; normalized y units. Query distance shares: '
      + ', '.join(f'{a:g}-{b:g}: {((dist >= a) & (dist < b)).mean():.0%}' for a, b in BINS))
for label, members in (('all tasks', (0, 1)), ('GP tasks', (1,)), ('MLP tasks', (0,))):
    print(f'\n== {label}')
    print(f'{"distance":>9s} | {"n":>4s} | pred std | actual RMS error | ratio (>1 over-confident) | 90% coverage | true noise std')
    for a, b in BINS:
        sel = (dist >= a) & (dist < b) & np.isin(gp, members)
        if sel.sum() < 20:
            continue
        rms = np.sqrt(np.mean((y_obs[sel] - m[sel]) ** 2))
        pred = np.sqrt(np.mean(s[sel] ** 2))
        cov = np.mean((y_obs[sel] >= lo[sel]) & (y_obs[sel] <= hi[sel]))
        print(f'{a:>4g}-{b:<4g} | {sel.sum():4d} | {pred:8.3f} | {rms:16.3f} | {rms / pred:25.2f} | {cov:12.2f} | {np.median(true_std[sel]):.3f}')

R = Path('../results/toy_regression')
for ds in ('linear_regression', 'gaps'):
    run = R/ds/'MDLM-Regress-HetGP50-Geometry-0.8M/va_vanilla/seed_0'
    cx = pd.read_csv(run/'D_test.csv').iloc[:, 0].to_numpy(float)
    grid = np.arange(-15, 15 + 1e-9, 0.2)
    zg = np.abs(normalize_by_context(np.append(cx, grid)[:, None], list(range(len(cx))))[len(cx):, 0])
    far = np.abs(grid[:, None] - cx[None]).min(1) > 1
    print(f'{ds} far-field grid points on the same scale: '
          + ', '.join(f'{a:g}-{b:g}: {((zg[far] >= a) & (zg[far] < b)).mean():.0%}' for a, b in BINS))
