"""Mode-1 reference: the Bayes posterior predictive of the PFN GP classification prior.

For one context (x, y) and query points, integrate the prior the checkpoint was
trained on by Monte Carlo: draw hyperparameters from the Gamma hyperpriors,
draw latent functions f jointly at context + query points from the GP
(Matern-2.5 with ARD lengthscales, plus the latent noise), weight each draw by
the Bernoulli likelihood of the context labels, p(y=1|x) = sigmoid(f), and
average. As in VUD, total = H[E p], aleatoric = E[H[p]] and epistemic =
total - aleatoric (bits).

The MDLM only sees x z-scored with its context rows (normalize_by_context).
Training contexts are uniform points in [0, 1], whose z-scores map back as
raw = 0.5 + z / sqrt(12); the query points use the same map.
"""
import math

import numpy as np
import torch

from small_mdlm.episodes import normalize_by_context


def matern52(a, b, lengthscale, outputscale):
    r = torch.cdist(a / lengthscale, b / lengthscale)
    s = math.sqrt(5) * r
    return outputscale * (1 + s + s ** 2 / 3) * torch.exp(-s)


def entropy_bits(p):
    p = p.clamp(1e-12, 1 - 1e-12)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


@torch.no_grad()
def posterior_predictive(x_context, y_context, x_query, outputscale_concentration=0.5, outputscale_rate=0.15,
                         lengthscale_concentration=3.0, lengthscale_rate=6.0, noise=1e-3,
                         hyper_draws=4000, function_draws=500, device='cpu', seed=0):
    x_context, x_query = np.asarray(x_context, float), np.asarray(x_query, float)
    n, d = x_context.shape
    z = normalize_by_context(np.vstack([x_context, x_query]), list(range(n)))
    active = z.std(0) > 0          # the zero x2 of 1-D tasks carries no information
    raw = torch.tensor(0.5 + z[:, active] / math.sqrt(12), dtype=torch.float64, device=device)
    y = torch.tensor(np.asarray(y_context, float), dtype=torch.float64, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    gamma = lambda k, rate, shape: torch._standard_gamma(torch.full(shape, k, dtype=torch.float64, device=device),
                                                         generator=gen) / rate
    points = raw.shape[0]
    sum_w, sum_wp, sum_wh = 0.0, torch.zeros(points - n, dtype=torch.float64, device=device), \
        torch.zeros(points - n, dtype=torch.float64, device=device)
    log_w_all = []
    for _ in range(hyper_draws):
        lengthscale = gamma(lengthscale_concentration, lengthscale_rate, (int(active.sum()),))
        outputscale = gamma(outputscale_concentration, outputscale_rate, ())
        k = matern52(raw, raw, lengthscale, outputscale) + (noise + 1e-8) * torch.eye(points, dtype=torch.float64, device=device)
        chol = torch.linalg.cholesky(k)
        f = chol @ torch.randn(points, function_draws, dtype=torch.float64, device=device, generator=gen)
        fc, fq = f[:n], f[n:]
        log_w = (y[:, None] * torch.nn.functional.logsigmoid(fc)
                 + (1 - y[:, None]) * torch.nn.functional.logsigmoid(-fc)).sum(0)
        log_w_all.append(log_w)
        w = torch.exp(log_w)            # likelihoods of 15 binary labels stay well above underflow
        p = torch.sigmoid(fq)
        sum_w += float(w.sum())
        sum_wp += p @ w
        sum_wh += entropy_bits(p) @ w
    log_w_all = torch.cat(log_w_all)
    weights = torch.exp(log_w_all - log_w_all.max())
    ess = float(weights.sum() ** 2 / (weights ** 2).sum())
    p1 = sum_wp / sum_w
    total = entropy_bits(p1)
    aleatoric = sum_wh / sum_w
    return {'p1': p1.cpu().numpy(), 'total_bits': total.cpu().numpy(), 'aleatoric_bits': aleatoric.cpu().numpy(),
            'epistemic_bits': (total - aleatoric).cpu().numpy(), 'effective_sample_size': ess}
