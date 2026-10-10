"""Regression prior members: heteroskedastic controlled GP and TabPFN v1's MLP prior in regression mode.

het_gp (a local prior, extending small_mdlm/vendor/controlled_gp_prior.py to regression):
  f ~ GP(0, RBF(l)), unit signal variance, l ~ logU[ls_min, ls_max] in the inputs' coordinates;
  noise std sigma0 from the controlled GP's regimes (clean: variance ratio logU[1e-6, 1e-3]; noisy: logU[0.03, 1];
  equally likely); with probability het_prob the noise varies smoothly: sigma(x) = sigma0 exp(a h(x)),
  a ~ U(0, a_max), h ~ GP(0, RBF(l_g)), l_g ~ logU[0.5, 3]. y = f(x) + sigma(x) eps.
mlp: TabPFN v1's MLP prior (small_mdlm/vendor/tabpfn_prior.py, non-causal) with the network's continuous output as
  y (v1's regression mode, flexible_categorical num_classes = 0); its noise comes from the hidden-layer noise.
  The returned mean/std per point are a Monte Carlo estimate over the noise.
Both return the true mean and noise std at every point, used only for validation metrics.
The exact-posterior screen of this prior is dllm-training/tmp/scripts/regression_reference.py.
"""
from dataclasses import dataclass
import math

import torch

from . import tabpfn_prior


@dataclass(frozen=True)
class HetGPConfig:
    ls_min: float = 0.1
    ls_max: float = 3.0
    lsg_min: float = 0.5
    lsg_max: float = 3.0
    clean_probability: float = 0.5
    clean_variance_ratio_min: float = 1e-6
    clean_variance_ratio_max: float = 1e-3
    noisy_variance_ratio_min: float = 0.03
    noisy_variance_ratio_max: float = 1.0
    het_prob: float = 0.5
    a_max: float = 3.0
    jitter: float = 1e-6


def _log_u(count, lo, hi, gen, device):
    u = torch.rand(count, device=device, generator=gen, dtype=torch.float64)
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)


@torch.no_grad()
def het_gp_at(x, cfg=HetGPConfig(), gen=None):
    """x (count, P, d) -> y, true mean f and noise std sigma, each (count, P); one task per row."""
    x = x.double()
    count, points, _ = x.shape
    dev = x.device
    ls = _log_u(count, cfg.ls_min, cfg.ls_max, gen, dev)
    lsg = _log_u(count, cfg.lsg_min, cfg.lsg_max, gen, dev)
    clean = torch.rand(count, device=dev, generator=gen) < cfg.clean_probability
    ratio = torch.where(clean, _log_u(count, cfg.clean_variance_ratio_min, cfg.clean_variance_ratio_max, gen, dev),
                        _log_u(count, cfg.noisy_variance_ratio_min, cfg.noisy_variance_ratio_max, gen, dev))
    het = torch.rand(count, device=dev, generator=gen) < cfg.het_prob
    a = torch.where(het, cfg.a_max * torch.rand(count, device=dev, generator=gen, dtype=torch.float64), torch.zeros_like(ratio))
    d2 = ((x[:, :, None] - x[:, None]) ** 2).sum(-1)
    eye = torch.eye(points, device=dev, dtype=x.dtype)
    kf = torch.exp(-0.5 * d2 / ls[:, None, None] ** 2) + cfg.jitter * eye
    kg = torch.exp(-0.5 * d2 / lsg[:, None, None] ** 2) + cfg.jitter * eye
    z = lambda: torch.randn(count, points, 1, device=dev, generator=gen, dtype=x.dtype)
    f = (torch.linalg.cholesky(kf) @ z()).squeeze(-1)
    h = (torch.linalg.cholesky(kg) @ z()).squeeze(-1)
    sigma = ratio.sqrt()[:, None] * torch.exp(a[:, None] * h)
    y = f + sigma * torch.randn(count, points, device=dev, generator=gen, dtype=x.dtype)
    return y, f, sigma


@torch.no_grad()
def mlp_at(net, x, gen=None, draws=16):
    """TabPFN v1 MLP regression: x (count, P, d) in each network's input scale -> one noisy y, and the Monte Carlo
    mean and std of the output over the hidden noise, each (count, P)."""
    y = tabpfn_prior.forward(net, x, gen)
    outs = torch.stack([tabpfn_prior.forward(net, x, gen) for _ in range(draws)], 0)
    return y, outs.mean(0), outs.std(0)
