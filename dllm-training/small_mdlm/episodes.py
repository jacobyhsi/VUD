"""Same-function contexts and multiple queries; separate from legacy data.py."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class EpisodeConfig:
    input_size: int = 1
    context_min: int = 3
    context_max: int = 30
    max_queries: int = 2
    input_range: float = 30.0
    weight_scale: float = 2.0
    bias_scale: float = 3.0
    noise_std: float = 2.0
    # Broad prior. All off by default, which reproduces the original episodes.
    # noise_std_max: per-episode noise std ~ log-uniform[noise_std, noise_std_max].
    noise_std_max: Optional[float] = None
    # cluster_prob: P(context x from one Gaussian cluster instead of uniform).
    # Centre ~ uniform over the middle half of the range, std ~ log-uniform.
    cluster_prob: float = 0.0
    cluster_std_min: float = 0.5
    cluster_std_max: float = 5.0
    # near_query_prob: P(the auxiliary row sits near the final query, like
    # VUD's perturbed z). The auxiliary is the hidden first query when there
    # are two queries, otherwise one observed context row.
    near_query_prob: float = 0.0
    near_query_std_min: float = 0.05
    near_query_std_max: float = 2.0
    # task 'classification': binary logistic labels, p(y=1|x) = sigmoid(w.(x - c)).
    # |w| ~ log-uniform[slope_min, slope_max] with a random direction (sign in 1-D),
    # and the boundary point c ~ uniform over [-boundary_range/2, boundary_range/2].
    # Softer slopes mean more label noise (aleatoric uncertainty).
    task: str = 'regression'
    slope_min: float = 0.05
    slope_max: float = 5.0
    boundary_range: float = 20.0
    # task 'classification_mix': a mixture of classification task families,
    # built in a unit box and mapped to x = scale * u + shift, where
    # scale ~ log-uniform[scale_min, scale_max] and shift ~ U(+-0.3 scale).
    # Families (probabilities family_linear/family_rff/family_curves):
    #   linear  - softmax(W u + b), random slopes (label noise from soft slopes)
    #   rff     - smooth random functions (GP-like, random length-scale) as logits
    #   curves  - generative: each class is a mixture of Gaussians along random
    #             arcs or at points (moons- and blob-like); p(y|x) by Bayes rule
    # one_d_prob: fraction of tasks that use only x1 (other features exactly 0),
    # which is how 1-D datasets are fed to a 2-D checkpoint.
    # task 'regression_mix': general regression on the same x range, mixing
    #   linear  - the broad linear prior above (weight_scale, bias_scale, noise range)
    #   modes   - 2-3 x clusters, each with its own line and noise level (or the
    #             shared line), like VUD's gaps; y for any x comes from a cluster
    #             drawn by its responsibility at that x
    #   smooth  - a line plus a smooth random function, with noise that may vary smoothly in x
    regression_linear: float = 0.35
    regression_modes: float = 0.4
    regression_smooth: float = 0.25
    # task 'regression_gp': general regression y = f(x) + sigma(x) * eps where,
    # per task, f is ONE fixed function: a linear trend (weight_scale,
    # bias_scale) plus, with the probabilities below, a random function drawn
    # like a Gaussian-process sample (random Fourier features) with an RBF
    # (smooth), periodic or Laplace (rough) kernel. Its length-scale is
    # log-uniform on [gp_length_min, gp_length_max] and amplitude log-uniform on
    # [gp_amplitude_min, gp_amplitude_max]. Base noise std is log-uniform on
    # [noise_std, noise_std_max]; with heteroscedastic_prob it varies smoothly
    # in x. Unobserved regions therefore have one unknown value (epistemic),
    # never a random choice between functions.
    gp_none: float = 0.3
    gp_rbf: float = 0.35
    gp_periodic: float = 0.15
    gp_rough: float = 0.2
    gp_length_min: float = 0.5
    gp_length_max: float = 15.0
    gp_amplitude_min: float = 0.3
    gp_amplitude_max: float = 15.0
    heteroscedastic_prob: float = 0.5
    # multi_cluster_prob: P(context x from 2-3 separated clusters, leaving gaps).
    multi_cluster_prob: float = 0.0
    # task 'classification_pfn': datasets from a pre-generated bank of the PFN
    # GP-mixture classification prior (generate_pfn_bank.py; x in [0, 1]^d).
    # Each episode takes a random subset of one dataset's points and z-scores x
    # with the context rows' mean/std, as PFNs normalize data. The last
    # bank_holdout datasets are reserved for validation/test.
    bank_path: Optional[str] = None
    bank_holdout: int = 20000
    # Opt-in fixed layouts from generate_tabpfn_episode_bank.py. Legacy banks
    # keep their original random subset sampling and random-number stream.
    pfn_fixed_episodes: bool = False
    # pfn_local_context_prob: P(context = the n points nearest a random centre,
    # queries drawn from the remaining, farther points). The prior is
    # unchanged; this only chooses which of a dataset's points are context, so
    # the model also sees queries beyond the context region (VUD sweeps x
    # far from the data). 0 keeps the random split and reproduces older runs.
    pfn_local_context_prob: float = 0.0
    num_classes: int = 2
    scale_min: float = 1.0
    scale_max: float = 12.0
    one_d_prob: float = 0.3
    family_linear: float = 0.25
    family_rff: float = 0.35
    family_curves: float = 0.4

    def __post_init__(self):
        if self.input_size < 1 or not 1 <= self.context_min <= self.context_max:
            raise ValueError('Invalid feature/context sizes')
        if self.max_queries not in (1, 2):
            raise ValueError('Prototype supports one or two queries')
        values = [self.input_range, self.weight_scale, self.bias_scale, self.noise_std]
        if not np.isfinite(values).all() or self.input_range <= 0 or min(values[1:]) < 0:
            raise ValueError('Invalid distribution scales')
        if self.noise_std_max is not None and not self.noise_std_max >= self.noise_std > 0:
            raise ValueError('Log-uniform noise needs 0 < noise_std <= noise_std_max')
        for p in (self.cluster_prob, self.near_query_prob):
            if not 0 <= p <= 1:
                raise ValueError('Probabilities must lie in [0, 1]')
        if self.task not in ('regression', 'regression_mix', 'regression_gp', 'regression_pfn',
                             'classification', 'classification_mix', 'classification_pfn'):
            raise ValueError(f'Unknown task {self.task!r}')
        if self.task == 'classification_pfn' and not self.bank_path:
            raise ValueError('classification_pfn needs bank_path')
        if self.pfn_fixed_episodes and self.task not in ('classification_pfn', 'regression_pfn'):
            raise ValueError('Fixed PFN episodes require classification_pfn or regression_pfn')
        if self.task == 'regression_pfn' and (not self.bank_path or not self.pfn_fixed_episodes):
            raise ValueError('regression_pfn needs bank_path and pfn_fixed_episodes (generate_regression_episode_bank.py)')
        if self.task == 'classification_mix':
            if self.input_size > 2 or not 2 <= self.num_classes <= 10:
                raise ValueError('classification_mix supports 1-2 inputs and 2-10 classes')
            if not 0 < self.scale_min <= self.scale_max or not 0 <= self.one_d_prob <= 1:
                raise ValueError('Invalid scale range or one_d_prob')
            if min(self.family_linear, self.family_rff, self.family_curves) < 0 or \
                    self.family_linear + self.family_rff + self.family_curves <= 0:
                raise ValueError('Family probabilities must be non-negative and not all zero')
        for lo, hi in ((self.cluster_std_min, self.cluster_std_max),
                       (self.near_query_std_min, self.near_query_std_max),
                       (self.slope_min, self.slope_max)):
            if not 0 < lo <= hi:
                raise ValueError('Log-uniform ranges need 0 < min <= max')

    @property
    def broad(self):
        return (self.noise_std_max is not None or self.cluster_prob > 0
                or self.near_query_prob > 0 or self.task != 'regression')


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def episode(config, seed, index, split='train'):
    split_id = {'train': 0, 'validation': 1, 'test': 2}[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, index]))
    n = int(rng.integers(config.context_min, config.context_max + 1))
    q = int(rng.integers(1, config.max_queries + 1))
    if config.task == 'classification':
        return _classification_episode(config, rng, n, q)
    if config.task == 'classification_mix':
        return _mixed_classification_episode(config, rng, n, q)
    if config.task == 'classification_pfn':
        return _pfn_classification_episode(config, rng, n, q, split)
    if config.task == 'regression_pfn':
        return _pfn_regression_episode(config, rng, split)
    if config.task == 'regression_mix':
        return _mixed_regression_episode(config, rng, n, q)
    if config.task == 'regression_gp':
        return _gp_regression_episode(config, rng, n, q)
    w = rng.normal(0, config.weight_scale, config.input_size)
    bias = rng.normal(0, config.bias_scale)
    if config.broad:
        return _broad_episode(config, rng, n, q, w, bias)
    x = rng.uniform(-config.input_range / 2, config.input_range / 2,
                    (n + q, config.input_size))
    y = x @ w + bias + rng.normal(0, config.noise_std, n + q)
    # Same layout for all rows: ordinary observations can serve as observed
    # auxiliaries. Randomly place a masked auxiliary among the demonstrations
    # (the VUD PP prompt shuffles that row into the context).
    targets = [n + q - 1]
    if q == 2:
        targets.insert(0, int(rng.integers(0, n + 1)))
    return {'x': x, 'y': y, 'target_rows': targets,
            'weights': w, 'bias': bias, 'noise_std': config.noise_std}


def _broad_episode(config, rng, n, q, w, bias):
    x, targets = _covariates(config, rng, n, q)
    sigma = (config.noise_std if config.noise_std_max is None
             else _log_uniform(rng, config.noise_std, config.noise_std_max))
    y = x @ w + bias + rng.normal(0, sigma, len(x))
    return {'x': x, 'y': y, 'target_rows': targets,
            'weights': w, 'bias': bias, 'noise_std': sigma}


def _classification_episode(config, rng, n, q):
    d = config.input_size
    direction = rng.normal(size=d)
    w = direction / np.linalg.norm(direction) * _log_uniform(rng, config.slope_min, config.slope_max)
    centre = rng.uniform(-config.boundary_range / 2, config.boundary_range / 2, d)
    x, targets = _covariates(config, rng, n, q)
    p = 1 / (1 + np.exp(-(x - centre) @ w))
    y = (rng.random(len(x)) < p).astype(float)
    return {'x': x, 'y': y, 'target_rows': targets, 'weights': w,
            'bias': float(-centre @ w), 'boundary': centre, 'p': p,
            'probs': np.stack([1 - p, p], 1)}


def _covariates(config, rng, n, q):
    """Context x (uniform or one cluster), a uniform query, and the auxiliary row."""
    d, half = config.input_size, config.input_range / 2
    # No random draw when multi-cluster layouts are off, so older configs reproduce.
    if config.multi_cluster_prob and rng.random() < config.multi_cluster_prob:
        k = int(rng.integers(2, 4))
        centres = rng.uniform(-0.8 * half, 0.8 * half, (k, d))
        spreads = np.exp(rng.uniform(np.log(0.3), np.log(2.0), k))
        which = rng.choice(k, size=n, p=rng.dirichlet(np.full(k, 2.0)))
        context = centres[which] + rng.normal(size=(n, d)) * spreads[which, None]
    elif rng.random() < config.cluster_prob:
        centre = rng.uniform(-half / 2, half / 2, d)
        spread = _log_uniform(rng, config.cluster_std_min, config.cluster_std_max)
        context = rng.normal(centre, spread, (n, d))
    else:
        context = rng.uniform(-half, half, (n, d))
    # Queries cover the whole range, including far from a clustered context.
    query = rng.uniform(-half, half, (1, d))
    aux = None
    if rng.random() < config.near_query_prob:
        near = query + rng.normal(0, _log_uniform(rng, config.near_query_std_min,
                                                  config.near_query_std_max), (1, d))
        if q == 2:
            aux = near
        else:
            context[rng.integers(n)] = near[0]
    elif q == 2:
        aux = rng.uniform(-half, half, (1, d))
    targets = [n + q - 1]
    if q == 2:
        # Place the hidden auxiliary among the demonstrations, as VUD's PP prompt does.
        position = int(rng.integers(0, n + 1))
        context = np.insert(context, position, aux[0], axis=0)
        targets.insert(0, position)
    return np.concatenate([context, query]), targets


def _softmax(logits):
    logits = logits - logits.max(-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(-1, keepdims=True)


def _unit_context(rng, n, d):
    """Context in the unit box: uniform, one cluster, or two to three clusters."""
    kind = rng.random()
    if kind < 0.4:
        return rng.uniform(-1, 1, (n, d))
    count = 1 if kind < 0.7 else int(rng.integers(2, 4))
    centres = rng.uniform(-0.6, 0.6, (count, d))
    spreads = np.exp(rng.uniform(np.log(0.05), np.log(0.5), count))
    which = rng.integers(count, size=n)
    return centres[which] + rng.normal(size=(n, d)) * spreads[which, None]


def _curve_components(rng, d, classes):
    """Per class, Gaussian means along a random arc (or a single point) with one std."""
    parts = []
    for label in range(classes):
        for _ in range(int(rng.integers(1, 3))):
            std = float(np.exp(rng.uniform(np.log(0.03), np.log(0.4))))
            centre = rng.uniform(-0.6, 0.6, d)
            if rng.random() < 0.5:
                means = centre[None]
            else:
                radius = float(np.exp(rng.uniform(np.log(0.2), np.log(1.0))))
                start, span = rng.uniform(0, 2 * np.pi), rng.uniform(0.5 * np.pi, 1.5 * np.pi)
                angles = start + span * np.linspace(0, 1, 24)
                arc = np.stack([np.cos(angles), np.sin(angles)], 1) * radius
                means = centre + arc[:, :d] if d > 1 else centre + arc[:, :1]
            parts.append((label, means, std))
    weights = rng.dirichlet(np.full(len(parts), 2.0))
    return parts, weights


def _curve_log_density(x, parts, weights, classes):
    """log p(x, y=k) for each class k under the curve mixture."""
    d = x.shape[1]
    out = np.full((len(x), classes), -np.inf)
    for (label, means, std), weight in zip(parts, weights):
        sq = ((x[:, None, :] - means[None]) ** 2).sum(-1)
        log_k = (np.log(weight) - np.log(len(means)) - 0.5 * d * np.log(2 * np.pi * std ** 2)
                 + np.logaddexp.reduce(-0.5 * sq / std ** 2, axis=1))
        out[:, label] = np.logaddexp(out[:, label], log_k)
    return out


def _mixed_classification_episode(config, rng, n, q):
    d_full, k = config.input_size, config.num_classes
    d = 1 if d_full > 1 and rng.random() < config.one_d_prob else d_full
    scale = _log_uniform(rng, config.scale_min, config.scale_max)
    shift = rng.uniform(-0.3, 0.3, d) * scale
    family = rng.choice(3, p=np.array([config.family_linear, config.family_rff, config.family_curves])
                        / (config.family_linear + config.family_rff + config.family_curves))
    if family == 2:
        parts, weights = _curve_components(rng, d, k)
        # Context drawn from the class mixture itself.
        pick = rng.choice(len(parts), size=n, p=weights)
        context = np.stack([parts[i][1][rng.integers(len(parts[i][1]))] for i in pick])
        context = context + rng.normal(size=(n, d)) * np.array([parts[i][2] for i in pick])[:, None]
        probs_of = lambda u: _softmax(_curve_log_density(u, parts, weights, k))
    else:
        context = _unit_context(rng, n, d)
        if family == 0:
            slope = _log_uniform(rng, 0.5, 30.0)
            w = rng.normal(size=(k, d))
            w *= slope / np.linalg.norm(w, axis=1, keepdims=True)
            b = -(w * rng.uniform(-0.6, 0.6, (k, d))).sum(1)
            probs_of = lambda u: _softmax(u @ w.T + b)
        else:
            m, length = 64, _log_uniform(rng, 0.15, 1.0)
            omega = rng.normal(size=(m, d)) / length
            phase = rng.uniform(0, 2 * np.pi, m)
            amplitude = _log_uniform(rng, 1.0, 30.0)
            coef = rng.normal(size=(m, k)) * amplitude * np.sqrt(2 / m)
            probs_of = lambda u: _softmax(np.cos(u @ omega.T + phase) @ coef)
    # VUD sweeps queries over the data region and beyond.
    query = rng.uniform(-1.2, 1.2, (1, d))
    aux = None
    if rng.random() < config.near_query_prob:
        near = query + rng.normal(0, _log_uniform(rng, 0.01, 0.2), (1, d))
        if q == 2:
            aux = near
        else:
            context[rng.integers(n)] = near[0]
    elif q == 2:
        aux = rng.uniform(-1.2, 1.2, (1, d))
    targets = [n + q - 1]
    if q == 2:
        position = int(rng.integers(0, n + 1))
        context = np.insert(context, position, aux[0], axis=0)
        targets.insert(0, position)
    u = np.concatenate([context, query])
    probs = probs_of(u)
    y = np.array([rng.choice(k, p=row) for row in probs], dtype=float)
    x = np.zeros((len(u), d_full))
    x[:, :d] = u * scale + shift
    return {'x': x, 'y': y, 'target_rows': targets, 'probs': probs, 'p': probs[:, 1],
            'family': ('linear', 'rff', 'curves')[family], 'active_dims': d}


def _mixed_regression_episode(config, rng, n, q):
    """General regression episode; returns the true mean and noise std of every row."""
    weights = np.array([config.regression_linear, config.regression_modes, config.regression_smooth])
    family = rng.choice(3, p=weights / weights.sum())
    d, half = config.input_size, config.input_range / 2
    noise = lambda: _log_uniform(rng, config.noise_std, config.noise_std_max or config.noise_std)
    line = lambda: (rng.normal(0, config.weight_scale, d), rng.normal(0, config.bias_scale))
    if family == 0:
        w, b = line()
        e = _broad_episode(config, rng, n, q, w, b)
        e.update(family='linear', mean=e['x'] @ w + b, std=np.full(len(e['x']), e['noise_std']))
        return e
    if family == 1:
        k = int(rng.integers(2, 4))
        centres = rng.uniform(-0.8 * half, 0.8 * half, (k, d))
        spreads = np.exp(rng.uniform(np.log(0.3), np.log(2.0), k))
        mix = rng.dirichlet(np.full(k, 2.0))
        shared = line()
        lines = [shared if rng.random() < 0.5 else line() for _ in range(k)]
        sigmas = np.array([noise() for _ in range(k)])
        which = rng.choice(k, size=n, p=mix)
        context = centres[which] + rng.normal(size=(n, d)) * spreads[which, None]

        def draw(x):
            # Cluster responsibilities at x decide which line generates y.
            log_r = (np.log(mix) - d * np.log(spreads)
                     - 0.5 * (((x[:, None] - centres[None]) / spreads[None, :, None]) ** 2).sum(-1))
            r = np.exp(log_r - log_r.max(1, keepdims=True))
            r /= r.sum(1, keepdims=True)
            means = np.stack([x @ w + b for w, b in lines], 1)
            mode = np.array([rng.choice(k, p=row) for row in r])
            mean = (r * means).sum(1)
            var = (r * (sigmas ** 2 + means ** 2)).sum(1) - mean ** 2
            return means[np.arange(len(x)), mode] + rng.normal(size=len(x)) * sigmas[mode], mean, np.sqrt(var)
    else:
        w, b = line()
        m, length = 64, _log_uniform(rng, 1.0, 10.0)
        omega, phase = rng.normal(size=(m, d)) / length, rng.uniform(0, 2 * np.pi, m)
        amplitude = _log_uniform(rng, 0.5, 10.0)
        coef = rng.normal(size=m) * amplitude * np.sqrt(2 / m)
        sigma0 = noise()
        wiggle = rng.normal(size=m) * np.sqrt(2 / m) * (rng.random() < 0.5)  # heteroscedastic half the time
        f = lambda x: x @ w + b + np.cos(x @ omega.T + phase) @ coef
        sd = lambda x: sigma0 * np.exp(np.clip(np.cos(x @ omega.T * 0.5 + phase) @ wiggle, -1.5, 1.5))
        context = None

        def draw(x):
            return f(x) + rng.normal(size=len(x)) * sd(x), f(x), sd(x)
    if context is None:
        x, targets = _covariates(config, rng, n, q)
    else:
        # Same query/auxiliary layout as the other families.
        query = rng.uniform(-half, half, (1, d))
        aux = None
        if rng.random() < config.near_query_prob:
            near = query + rng.normal(0, _log_uniform(rng, config.near_query_std_min,
                                                      config.near_query_std_max), (1, d))
            if q == 2:
                aux = near
            else:
                context[rng.integers(n)] = near[0]
        elif q == 2:
            aux = rng.uniform(-half, half, (1, d))
        targets = [n + q - 1]
        if q == 2:
            position = int(rng.integers(0, n + 1))
            context = np.insert(context, position, aux[0], axis=0)
            targets.insert(0, position)
        x = np.concatenate([context, query])
    y, mean, std = draw(x)
    return {'x': x, 'y': y, 'target_rows': targets, 'mean': mean, 'std': std,
            'family': ('linear', 'modes', 'smooth')[family]}


def _random_function(rng, d, kind, length, amplitude, m=64):
    """A random-Fourier-feature sample: RBF (Gaussian frequencies), rough
    (Laplace kernel, Cauchy frequencies) or periodic (harmonics of one period)."""
    phase = rng.uniform(0, 2 * np.pi, m)
    if kind == 'periodic':
        harmonics = rng.integers(1, 7, (m, 1))
        direction = rng.normal(size=(1, d))
        omega = 2 * np.pi * harmonics * direction / np.linalg.norm(direction) / length
        weight = rng.normal(size=m) * np.exp(-0.5 * harmonics[:, 0])
        weight *= amplitude / np.sqrt((weight ** 2).sum() / 2)
    else:
        scale = rng.standard_cauchy((m, d)) if kind == 'rough' else rng.normal(size=(m, d))
        omega = scale / length
        weight = rng.normal(size=m) * amplitude * np.sqrt(2 / m)
    return lambda x: np.cos(x @ omega.T + phase) @ weight


def _gp_regression_episode(config, rng, n, q):
    """y = f(x) + sigma(x) eps with one fixed f per task (see EpisodeConfig)."""
    d = config.input_size
    x, targets = _covariates(config, rng, n, q)
    w, b = rng.normal(0, config.weight_scale, d), rng.normal(0, config.bias_scale)
    kinds = np.array([config.gp_none, config.gp_rbf, config.gp_periodic, config.gp_rough])
    kind = ('none', 'rbf', 'periodic', 'rough')[rng.choice(4, p=kinds / kinds.sum())]
    mean = x @ w + b
    if kind != 'none':
        g = _random_function(rng, d, kind, _log_uniform(rng, config.gp_length_min, config.gp_length_max),
                             _log_uniform(rng, config.gp_amplitude_min, config.gp_amplitude_max))
        mean = mean + g(x)
    sigma0 = _log_uniform(rng, config.noise_std, config.noise_std_max or config.noise_std)
    std = np.full(len(x), sigma0)
    if rng.random() < config.heteroscedastic_prob:
        h = _random_function(rng, d, 'rbf', _log_uniform(rng, config.gp_length_min, config.gp_length_max), 1.0)
        std = sigma0 * np.exp(np.clip(h(x), -1.5, 1.5))
    y = mean + std * rng.normal(size=len(x))
    return {'x': x, 'y': y, 'target_rows': targets, 'mean': mean, 'std': std, 'family': kind}


def normalize_by_context(x, observed_rows, clip=50.0):
    """z-score each feature with the observed (context) rows' mean and std.

    Constant features, such as the zero x2 of 1-D tasks, stay at 0. Values are
    clipped so tiny contexts cannot produce numbers outside the tokenizer range.
    """
    x = np.asarray(x, dtype=float)
    context = x[observed_rows]
    std = context.std(0)
    z = (x - context.mean(0)) / np.where(std > 1e-8, std, 1.0)
    z[:, std <= 1e-8] = 0.0
    return np.clip(z, -clip, clip)


@lru_cache(maxsize=2)
def _load_bank(path):
    root = Path(path) if Path(path).is_absolute() else Path(__file__).resolve().parents[1]/path
    names = ['x', 'y', 'mean', 'std'] if (root/'mean.npy').exists() else ['x', 'latent', 'y']
    if (root/'context_size.npy').exists():
        if not (root/'meta.json').exists():
            raise ValueError('Fixed episode bank is incomplete')
        names += ['context_size', 'query_count']
    return {name: np.load(root/f'{name}.npy', mmap_mode='r') for name in names}


def normalize_targets(y, observed_rows, clip=50.0):
    """z-score y with the observed rows' mean and std (1 when constant); returns (z, mean, std)."""
    y = np.asarray(y, dtype=float)
    context = y[observed_rows]
    mean, std = float(context.mean()), float(context.std())
    std = std if std > 1e-8 else 1.0
    return np.clip((y - mean) / std, -clip, clip), mean, std


def _pfn_regression_episode(config, rng, split):
    """A fixed regression episode (generate_regression_episode_bank.py); x and y are z-scored with the context."""
    bank = _load_bank(config.bank_path)
    if 'context_size' not in bank or 'mean' not in bank:
        raise ValueError('regression_pfn needs a fixed regression episode bank')
    total, seq_len = bank['y'].shape
    hold = config.bank_holdout
    if not 0 < hold < total:
        raise ValueError('Bank too small for the holdout')
    lo, hi = {'train': (0, total - hold), 'validation': (total - hold, total - hold // 2),
              'test': (total - hold // 2, total)}[split]
    i = int(rng.integers(lo, hi))
    n, q = int(bank['context_size'][i]), int(bank['query_count'][i])
    if not config.context_min <= n <= config.context_max or not 1 <= q <= config.max_queries or n + q > seq_len:
        raise ValueError('Fixed episode layout does not match configuration')
    order = np.concatenate([rng.permutation(n), np.arange(n, n + q)])
    rows = list(range(n)) + [n + q - 1]
    targets = [n]
    if q == 2:
        position = int(rng.integers(0, n + 1))
        rows.insert(position, n)
        targets = [position, n + 1]
    pick = order[rows]
    x = np.asarray(bank['x'][i], dtype=float)[pick]
    y, mean, std = (np.asarray(bank[k][i], dtype=float)[pick] for k in ('y', 'mean', 'std'))
    observed = [r for r in range(len(rows)) if r not in targets]
    y_z, y_mean, y_std = normalize_targets(y, observed)
    return {'x': normalize_by_context(x, observed), 'y': y_z, 'target_rows': targets,
            'mean': (mean - y_mean) / y_std, 'std': std / y_std, 'bank_index': i}


def _pfn_classification_episode(config, rng, n, q, split):
    bank = _load_bank(config.bank_path)
    if ('context_size' in bank) != config.pfn_fixed_episodes:
        raise ValueError('Bank layout and pfn_fixed_episodes setting disagree')
    total, seq_len = bank['y'].shape
    hold = config.bank_holdout
    if not 0 < hold < total or n + q > seq_len:
        raise ValueError('Bank too small for the holdout or context size')
    lo, hi = {'train': (0, total - hold), 'validation': (total - hold, total - hold // 2),
              'test': (total - hold // 2, total)}[split]
    i = int(rng.integers(lo, hi))
    if config.pfn_fixed_episodes:
        if config.pfn_local_context_prob:
            raise ValueError('Fixed episodes cannot be combined with local-context resampling')
        if 'context_size' not in bank:
            raise ValueError('Fixed episodes need context_size/query_count metadata')
        n, q = int(bank['context_size'][i]), int(bank['query_count'][i])
        if not config.context_min <= n <= config.context_max or not 1 <= q <= config.max_queries or n + q > seq_len:
            raise ValueError('Fixed episode layout does not match configuration')
        # Shuffle observations but preserve the designated nearby auxiliary and
        # final query. Padding rows are never used as context or targets.
        order = np.concatenate([rng.permutation(n), np.arange(n, n + q)])
    elif config.pfn_local_context_prob and rng.random() < config.pfn_local_context_prob:
        points = np.asarray(bank['x'][i], dtype=float)
        centre = points[rng.integers(seq_len)]
        nearest = np.argsort(((points - centre) ** 2).sum(1), kind='stable')
        order = np.concatenate([rng.permutation(nearest[:n]), rng.permutation(nearest[n:])[:q]])
    else:
        order = rng.permutation(seq_len)[:n + q]
    x = np.asarray(bank['x'][i], dtype=float)[order]
    latent = np.asarray(bank['latent'][i], dtype=float)[order]
    y = np.asarray(bank['y'][i], dtype=float)[order]
    rows = list(range(n)) + [n + q - 1]
    targets = [n]
    if q == 2:
        # Hidden auxiliary placed among the demonstrations, as elsewhere.
        position = int(rng.integers(0, n + 1))
        rows.insert(position, n)
        targets = [position, n + 1]
    else:
        targets = [n]
    x, latent, y = x[rows], latent[rows], y[rows]
    observed = [r for r in range(len(rows)) if r not in targets]
    p = 1 / (1 + np.exp(-latent))
    return {'x': normalize_by_context(x, observed), 'y': y, 'target_rows': targets,
            'probs': np.stack([1 - p, p], 1), 'p': p, 'bank_index': i}
