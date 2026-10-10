"""Paired IID / VUD-geometry episodes with unchanged TabPFN network hyperpriors.

With gp_fraction > 0, that share of tasks comes from the controlled GP prior
(small_mdlm/vendor/controlled_gp_prior.py) instead, with the same IID N(0,1)
calibration inputs and the same extra-location geometry. Its threshold does not
depend on the data, so one GP function drawn jointly at the base, control and
altered locations gives exact labels at the extra points (shared observation
noise at the two extra rows keeps the arms' common random numbers).

layout_prob > 0 replaces a task's IID N(0,1) input draws (all 36 points, so the
IID-arm extra points follow the same layout) with a structured layout: Gaussian
clusters, points along a random smooth curve, or a correlated Gaussian, each
standardized per task and feature. This only changes where inputs fall; the
prior over functions and the exact posterior given the inputs are unchanged.

The original 34 IID points calibrate the rank/value threshold. Extra locations
are evaluated with that SAME network and threshold, with fresh observation
noise. This extends the finite-dataset label generator, not an exact replica
of its original joint distribution; the paired IID control uses this same
extension. No evaluation dataset, labels, or posterior reference is used.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .vendor import tabpfn_prior as prior
from .vendor.controlled_gp_prior import ControlledGPConfig, sample_at_inputs, sample_hyperparameters


@dataclass(frozen=True)
class GeometryConfig:
    context_min: int = 3
    context_max: int = 32
    calibration_size: int = 34
    one_d_prob: float = 0.3
    broad_query_prob: float = 0.5
    query_range: float = 6.0
    near_query_prob: float = 0.7
    near_query_std: float = 0.1
    layout_prob: float = 0.0
    draws: int = 64

    def __post_init__(self):
        if not 3 <= self.context_min <= self.context_max <= self.calibration_size:
            raise ValueError('Need 3 <= context_min <= context_max <= calibration_size')
        if any(not 0 <= p <= 1 for p in (self.one_d_prob, self.broad_query_prob, self.near_query_prob, self.layout_prob)):
            raise ValueError('Invalid probability')
        if not np.isfinite([self.query_range, self.near_query_std]).all() or min(self.query_range, self.near_query_std) <= 0 or self.draws < 2:
            raise ValueError('Invalid geometry or probability-estimation settings')


def _layout_points(count, points, d, prob, gen, device):
    """Standard-normal-scale inputs: IID N(0,1), or with probability prob a structured layout (equally often
    Gaussian clusters, a random smooth curve, or a correlated Gaussian), standardized per task and feature."""
    iid = torch.randn(count, points, d, device=device, generator=gen)
    u = lambda *shape: torch.rand(*shape, device=device, generator=gen)
    n = lambda *shape: torch.randn(*shape, device=device, generator=gen)
    log_u = lambda lo, hi, *shape: torch.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u(*shape))
    # Clusters: 2-5 centres ~ N(0, 1.5^2), each point in a random cluster, spread log-uniform on [0.05, 0.5].
    k = torch.randint(2, 6, (count, 1), device=device, generator=gen)
    member = (u(count, points) * k).long()
    clusters = (1.5 * n(count, 5, d)).gather(1, member[..., None].expand(-1, -1, d)) + log_u(.05, .5, count, 1, 1) * n(count, points, d)
    # Curve: x(t) = a t + b sin(pi w t + phi), t ~ U(-1, 1), plus jitter log-uniform on [0.01, 0.2].
    t = 2 * u(count, points, 1) - 1
    curve = (n(count, 1, d) * t + n(count, 1, d) * torch.sin(np.pi * (1 + 3 * u(count, 1, 1)) * t + 2 * np.pi * u(count, 1, d))
             + log_u(.01, .2, count, 1, 1) * n(count, points, d))
    # Correlated Gaussian: IID points through a random linear map.
    correlated = n(count, points, d) @ n(count, d, d)
    kind = torch.randint(3, (count,), device=device, generator=gen)
    layout = torch.where((kind == 0)[:, None, None], clusters, torch.where((kind == 1)[:, None, None], curve, correlated))
    layout = (layout - layout.mean(1, keepdim=True)) / (layout.std(1, keepdim=True) + 1e-6)
    return torch.where((u(count) < prob)[:, None, None], layout, iid)


def _extra_locations(control, base, n, cfg, rng):
    """Control (auxiliary, final query) is IID. Intervention changes query/auxiliary covariates only."""
    altered = control.clone()
    broad = rng.random(len(n)) < cfg.broad_query_prob
    near = rng.random(len(n)) < cfg.near_query_prob
    counts = torch.as_tensor(n, device=base.device)
    mask = (torch.arange(base.shape[1], device=base.device)[None] < counts[:, None])[..., None]
    mean = (base * mask).sum(1) / counts[:, None]
    std = (((base-mean[:, None]).square() * mask).sum(1) / counts[:, None]).sqrt()
    offset = torch.as_tensor(rng.uniform(-cfg.query_range, cfg.query_range, (len(n), base.shape[-1])), device=base.device, dtype=base.dtype)
    altered[:, 1] = torch.where(torch.as_tensor(broad, device=base.device)[:, None], mean + std * offset, altered[:, 1])
    offset = torch.as_tensor(rng.normal(0, cfg.near_query_std, (len(n), base.shape[-1])), device=base.device, dtype=base.dtype)
    altered[:, 0] = torch.where(torch.as_tensor(near, device=base.device)[:, None], altered[:, 1] + std * offset, altered[:, 0])
    return control, altered, broad, near


def _fill(arms, todo, ok, n, q, d, cfg, extras, broad, near):
    """Write accepted tasks into both arms. extras: {'iid'|'geometry': (xs, ys, latent)}, all (tasks, 36, ...)."""
    for name, (xs, ys, latent) in extras.items():
        for j in np.flatnonzero(ok):
            row = todo[j]
            obs = int(n[row])
            # Preserve D-size across arms. In single-query episodes an
            # auxiliary replaces one ordinary observed point.
            order = list(range(obs - (q[row] == 1))) + [cfg.calibration_size, cfg.calibration_size + 1]
            length = obs + int(q[row])
            arms[name]['x'][row, :length, :d] = xs[j, order]
            arms[name]['y'][row, :length] = ys[j, order]
            arms[name]['latent'][row, :length] = latent[j, order]
            if name == 'geometry':
                arms[name]['broad_query'][row] = broad[j]
                arms[name]['near_query'][row] = near[j]


@torch.no_grad()
def _controlled_gp_tasks(arms, todo, n, q, d, cfg, gp_cfg, rng, gen, device):
    count = len(todo)
    if cfg.layout_prob > 0:
        points = _layout_points(count, cfg.calibration_size + 2, d, cfg.layout_prob, gen, device)
        base, control = points[:, :cfg.calibration_size], points[:, cfg.calibration_size:]
    else:  # the original draws, so layout-free banks stay reproducible
        base = torch.randn(count, cfg.calibration_size, d, device=device, generator=gen)
        control = torch.randn(count, 2, d, device=device, generator=gen)
    control, altered, broad, near = _extra_locations(control, base, n[todo], cfg, rng)
    hp = sample_hyperparameters(count, gp_cfg, gen, device)
    _, _, draw = sample_at_inputs(torch.cat([base, control, altered], 1), gp_cfg, gen, hp)
    f, t = draw['function_values'], hp['threshold'][:, None]
    sd = hp['noise_variance'].sqrt()[:, None]
    base_noise = sd * torch.randn(count, cfg.calibration_size, device=device, generator=gen, dtype=f.dtype)
    extra_noise = sd * torch.randn(count, 2, device=device, generator=gen, dtype=f.dtype)   # shared by both arms
    k = cfg.calibration_size
    extras = {}
    for name, cols, loc in (('iid', slice(k, k + 2), control), ('geometry', slice(k + 2, k + 4), altered)):
        fx = torch.cat([f[:, :k], f[:, cols]], 1)
        ys = (fx + torch.cat([base_noise, extra_noise], 1) > t).cpu().numpy()
        p = torch.special.ndtr((fx - t) / sd).clamp(1e-6, 1 - 1e-6)
        extras[name] = (torch.cat([base, loc], 1).cpu().numpy(), ys, torch.logit(p).float().cpu().numpy())
    ok = torch.isfinite(f).all(1).cpu().numpy()
    _fill(arms, todo, ok, n, q, d, cfg, extras, broad, near)
    return todo[~ok]


@torch.no_grad()
def make_paired_chunk(index, count, seed, cfg=GeometryConfig(), device='cpu', gp_fraction=0.0, gp_cfg=ControlledGPConfig()):
    """Fixed-layout banks: first n observed rows, then q hidden rows.

    For q=1, the final observed row is the auxiliary; for q=2 it is hidden.
    Padding beyond n+q is ignored by the loader. Both arms share all ordinary
    context observations, latent tasks, episode sizes and random noise draws.
    gp_fraction of the tasks come from the controlled GP prior (0 keeps old banks reproducible).
    """
    rng = np.random.default_rng([seed, index])
    gen = torch.Generator(device=device).manual_seed(seed * 1_000_003 + index)
    n = rng.integers(cfg.context_min, cfg.context_max + 1, count, dtype=np.int16)
    q = rng.integers(1, 3, count, dtype=np.int8)
    dims = np.where(rng.random(count) < cfg.one_d_prob, 1, 2).astype(np.int8)
    width = cfg.context_max + 2
    arms = {name: {'x': np.zeros((count, width, 2), np.float32),
                   'y': np.zeros((count, width), np.int8),
                   'latent': np.zeros((count, width), np.float32),
                   'context_size': n.copy(), 'query_count': q.copy(), 'dims': dims.copy(),
                   'broad_query': np.zeros(count, np.int8), 'near_query': np.zeros(count, np.int8)}
            for name in ('iid', 'geometry')}
    gp = rng.random(count) < gp_fraction if gp_fraction > 0 else np.zeros(count, bool)
    for d in (1, 2):
        todo = np.flatnonzero((dims == d) & gp)
        for _ in range(100):
            if not len(todo):
                break
            todo = _controlled_gp_tasks(arms, todo, n, q, d, cfg, gp_cfg, rng, gen, device)
        if len(todo):
            raise RuntimeError('Too many non-finite controlled-GP tasks')
        todo = np.flatnonzero((dims == d) & ~gp)
        attempts = 0
        while len(todo):
            attempts += 1
            if attempts > 100:
                raise RuntimeError('Too many non-finite TabPFN tasks')
            net = prior.sample_networks(len(todo), d, gen, device)
            if cfg.layout_prob > 0:
                z = _layout_points(len(todo), cfg.calibration_size + 2, d, cfg.layout_prob, gen, device)
                points = net['x_mean'][:, None] + net['x_std'][:, None] * z     # each network's input law
                base, layout_control = points[:, :cfg.calibration_size], points[:, cfg.calibration_size:]
            else:
                base = prior.sample_inputs(net, cfg.calibration_size, gen)
            base_out = prior.forward(net, base, gen)
            threshold = prior.threshold(net, base_out, gen)
            control = layout_control if cfg.layout_prob > 0 else prior.sample_inputs(net, 2, gen)  # auxiliary, final query
            control, altered, broad, near = _extra_locations(control, base, n[todo], cfg, rng)
            # Common random numbers: identical inputs produce identical noisy
            # labels in both arms. Noise remains independent between rows.
            state = gen.get_state()
            control_out = prior.forward(net, control, gen)
            gen.set_state(state)
            altered_out = prior.forward(net, altered, gen)
            state = gen.get_state()
            cp = prior.label_probs(net, torch.cat([base, control], 1), threshold, cfg.draws, gen)
            gen.set_state(state)
            ap = prior.label_probs(net, torch.cat([base, altered], 1), threshold, cfg.draws, gen)
            ok = (torch.isfinite(base).all((1, 2)) & torch.isfinite(base_out).all(1)
                  & torch.isfinite(threshold) & torch.isfinite(control_out).all(1)
                  & torch.isfinite(altered_out).all(1)
                  & torch.isfinite(control).all((1, 2)) & torch.isfinite(altered).all((1, 2)))
            extras = {}
            for name, extra, output, p in [('iid', control, control_out, cp), ('geometry', altered, altered_out, ap)]:
                p = p.clamp(0.5 / cfg.draws, 1 - 0.5 / cfg.draws)
                extras[name] = (torch.cat([base, extra], 1).cpu().numpy(),
                                (torch.cat([base_out, output], 1) > threshold[:, None]).cpu().numpy(),
                                torch.logit(p).cpu().numpy())
            _fill(arms, todo, ok.cpu().numpy(), n, q, d, cfg, extras, broad, near)
            todo = todo[~ok.cpu().numpy()]
    return arms
