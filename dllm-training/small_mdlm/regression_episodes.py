"""Fixed-layout regression episodes with VUD geometry, the regression analogue of tabpfn_episodes.py.

Each task: a prior member (het GP with probability gp_fraction, else TabPFN v1 MLP in regression mode), 34 base
inputs (IID N(0,1) or, with probability layout_prob, a structured layout; MLP inputs pass through each network's
input law), plus an auxiliary and a final query placed as in tabpfn_episodes (GeometryConfig: the query far from
the context with broad_query_prob, the auxiliary next to it with near_query_prob). One function is drawn jointly
at all points, so targets at the extra locations are exact. Layout as in tabpfn_episodes: first n observed rows
(for one query the auxiliary is the last of them), then the hidden rows. Inputs are 1-D.
"""
import numpy as np
import torch

from .tabpfn_episodes import GeometryConfig, _extra_locations, _layout_points
from .vendor import tabpfn_prior as prior
from .vendor.regression_priors import HetGPConfig, het_gp_at, mlp_at


@torch.no_grad()
def make_regression_chunk(index, count, seed, cfg=GeometryConfig(), device='cpu', gp_fraction=0.5, gp_cfg=HetGPConfig()):
    rng = np.random.default_rng([seed, index])
    gen = torch.Generator(device=device).manual_seed(seed * 1_000_003 + index)
    n = rng.integers(cfg.context_min, cfg.context_max + 1, count, dtype=np.int16)
    q = rng.integers(1, 3, count, dtype=np.int8)
    gp = rng.random(count) < gp_fraction
    width, k = cfg.context_max + 2, cfg.calibration_size
    out = {'x': np.zeros((count, width, 1), np.float32), 'y': np.zeros((count, width), np.float32),
           'mean': np.zeros((count, width), np.float32), 'std': np.zeros((count, width), np.float32),
           'context_size': n.copy(), 'query_count': q.copy(), 'gp_task': gp.astype(np.int8),
           'broad_query': np.zeros(count, np.int8), 'near_query': np.zeros(count, np.int8)}
    for member in (True, False):
        todo = np.flatnonzero(gp == member)
        for _ in range(100):
            if not len(todo):
                break
            z = _layout_points(len(todo), k + 2, 1, cfg.layout_prob, gen, device)
            if member:
                base, control = z[:, :k], z[:, k:]
            else:
                net = prior.sample_networks(len(todo), 1, gen, device)
                pts = net['x_mean'][:, None] + net['x_std'][:, None] * z
                base, control = pts[:, :k], pts[:, k:]
            _, altered, broad, near = _extra_locations(control, base, n[todo], cfg, rng)
            x = torch.cat([base, altered], 1)
            if member:
                y, mean, std = het_gp_at(x, gp_cfg, gen)
            else:
                y, mean, std = mlp_at(net, x, gen)
            ok = (torch.isfinite(y).all(1) & torch.isfinite(mean).all(1) & torch.isfinite(std).all(1)
                  & torch.isfinite(x).all(2).all(1)).cpu().numpy()
            xs, ys, ms, ss = (t.cpu().numpy() for t in (x, y, mean, std))
            for j in np.flatnonzero(ok):
                row, obs = todo[j], int(n[todo[j]])
                order = list(range(obs - (q[row] == 1))) + [k, k + 1]
                length = obs + int(q[row])
                out['x'][row, :length, 0] = xs[j, order, 0]
                out['y'][row, :length], out['mean'][row, :length], out['std'][row, :length] = ys[j, order], ms[j, order], ss[j, order]
                out['broad_query'][row], out['near_query'][row] = broad[j], near[j]
            todo = todo[~ok]
        if len(todo):
            raise RuntimeError('Too many non-finite regression tasks')
    return out
