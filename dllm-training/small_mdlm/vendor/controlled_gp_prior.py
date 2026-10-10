"""Opt-in smooth GP classification prior; a local adaptation, not TabPFN v1.

One task draws a lengthscale and an observation-noise/signal VARIANCE ratio.
The signal variance is one. A threshold is sampled independently of the
locations and observations, so labels at additional locations use the same
function and threshold; no dataset-dependent threshold is recalibrated.
The clean/noisy regimes describe observation noise, not posterior uncertainty.
No VUD datasets or labels are consulted by this generator.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ControlledGPConfig:
    lengthscale_min: float = 0.5
    lengthscale_max: float = 3.0
    clean_probability: float = 0.5
    clean_variance_ratio_min: float = 1e-6
    clean_variance_ratio_max: float = 1e-3
    noisy_variance_ratio_min: float = 0.03
    noisy_variance_ratio_max: float = 1.0
    threshold_std: float = 0.5
    jitter: float = 1e-8

    def __post_init__(self):
        ranges = ((self.lengthscale_min, self.lengthscale_max),
                  (self.clean_variance_ratio_min, self.clean_variance_ratio_max),
                  (self.noisy_variance_ratio_min, self.noisy_variance_ratio_max))
        if any(not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi)
               for lo, hi in ranges):
            raise ValueError('Lengthscale and variance-ratio ranges must be finite and positive')
        if not 0 <= self.clean_probability <= 1:
            raise ValueError('clean_probability must lie in [0, 1]')
        if not math.isfinite(self.threshold_std) or self.threshold_std < 0:
            raise ValueError('threshold_std must be finite and nonnegative')
        if not math.isfinite(self.jitter) or not 0 < self.jitter <= 1e-3:
            raise ValueError('Need 0 < jitter <= 1e-3')


def _log_uniform(count, lo, hi, generator, device):
    u = torch.rand(count, device=device, generator=generator, dtype=torch.float64)
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)


def sample_hyperparameters(count, config=ControlledGPConfig(), generator=None, device='cpu'):
    """Explicit per-task settings; noise_variance equals the variance ratio."""
    if count < 1:
        raise ValueError('count must be positive')
    clean = torch.rand(count, device=device, generator=generator) < config.clean_probability
    clean_ratio = _log_uniform(count, config.clean_variance_ratio_min,
                               config.clean_variance_ratio_max, generator, device)
    noisy_ratio = _log_uniform(count, config.noisy_variance_ratio_min,
                               config.noisy_variance_ratio_max, generator, device)
    return {
        'lengthscale': _log_uniform(count, config.lengthscale_min, config.lengthscale_max, generator, device),
        'noise_variance': torch.where(clean, clean_ratio, noisy_ratio),
        'clean': clean,
        'threshold': config.threshold_std * torch.randn(count, device=device, generator=generator, dtype=torch.float64),
    }


@torch.no_grad()
def sample_at_inputs(x, config=ControlledGPConfig(), generator=None, hyperparameters=None):
    """Jointly draw ONE GP at x (tasks, points, features), then independent labels.

    Returns fixed-task latent probabilities as logits, labels, and diagnostic
    draws. These are not posterior predictive probabilities given a context.
    Call once on the union of locations when correlated outputs are needed.
    """
    if x.ndim != 3 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError('Need finite inputs with shape (tasks, points, features)')
    count, points, _ = x.shape
    x = x.double()
    hp = (sample_hyperparameters(count, config, generator, x.device)
          if hyperparameters is None else hyperparameters)
    for key in ('lengthscale', 'noise_variance', 'threshold'):
        if hp[key].shape != (count,) or hp[key].device != x.device or not torch.isfinite(hp[key]).all():
            raise ValueError('Hyperparameter shapes/devices must match inputs')
    if (hp['lengthscale'] <= 0).any() or (hp['noise_variance'] <= 0).any():
        raise ValueError('Positive lengthscales and noise variances required')
    distance2 = ((x[:, :, None] - x[:, None]) ** 2).sum(-1)
    kernel = torch.exp(-0.5 * distance2 / hp['lengthscale'][:, None, None] ** 2)
    # Preserve unit marginal variance, including the numerical nugget.
    kernel = (kernel + config.jitter * torch.eye(points, device=x.device, dtype=x.dtype)) / (1 + config.jitter)
    f = (torch.linalg.cholesky(kernel) @ torch.randn(count, points, 1, device=x.device,
                                                   generator=generator, dtype=x.dtype)).squeeze(-1)
    sd = hp['noise_variance'].sqrt()[:, None]
    p = torch.special.ndtr((f - hp['threshold'][:, None]) / sd)
    observed = f + sd * torch.randn(count, points, device=x.device, generator=generator, dtype=x.dtype)
    y = (observed > hp['threshold'][:, None]).float()
    diagnostics = {**hp, 'function_values': f, 'probabilities': p}
    return torch.logit(p.clamp(1e-6, 1 - 1e-6)).float(), y, diagnostics


@torch.no_grad()
def sample_classification_datasets(count, seq_len, num_features, draws=None, generator=None,
                                   device='cpu', config=ControlledGPConfig(), return_diagnostics=False):
    """IID N(0,1) inputs, smooth GP functions, independent probit label noise.

    `draws` is unused; included for the existing prior-bag sampler interface.
    Label probabilities are analytic conditional on the sampled task.
    """
    if min(count, seq_len, num_features) < 1:
        raise ValueError('Positive dataset, point, and feature counts required')
    x = torch.randn(count, seq_len, num_features, device=device, generator=generator)
    latent, y, diagnostics = sample_at_inputs(x, config, generator)
    result = (x, latent, y)
    return (*result, diagnostics) if return_diagnostics else result
