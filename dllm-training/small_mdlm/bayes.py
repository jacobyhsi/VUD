"""Bayes oracle: the exact posterior predictive under the linear training prior.

Prior: w ~ N(0, weight_scale^2 I), b ~ N(0, bias_scale^2) and noise std sigma
either fixed or log-uniform on [noise_std, noise_std_max]. Given sigma, (w, b)
is conjugate-Gaussian; the log-uniform sigma is integrated on a grid that is
uniform in log(sigma), so every grid point has equal prior weight.

For a query x the predictive p(y | x, D) is a mixture of Gaussians with
  total variance = E[sigma^2 | D] + Var[w x + b | D]
so, as in VUD, aleatoric = E[sigma^2 | D] and epistemic = total - aleatoric.
The covariate distribution does not enter: it carries no information about
(w, b, sigma). Rounding of the tokenized values is ignored.
"""
import numpy as np


def noise_grid(config, points=128):
    if config.noise_std_max is None or config.noise_std_max == config.noise_std:
        return np.array([config.noise_std], dtype=float)
    return np.exp(np.linspace(np.log(config.noise_std), np.log(config.noise_std_max), points))


def _design(x, config):
    x = np.asarray(x, dtype=float).reshape(len(x), config.input_size)
    return np.column_stack([x, np.ones(len(x))]) if config.bias_scale > 0 else x


def posterior_predictive(config, x_context, y_context, x_query, points=128):
    """Posterior predictive at each query row: mean, total/aleatoric/epistemic variance."""
    phi, phi_q = _design(x_context, config), _design(x_query, config)
    y = np.asarray(y_context, dtype=float)
    prior_var = np.full(phi.shape[1], config.weight_scale ** 2)
    if config.bias_scale > 0:
        prior_var[-1] = config.bias_scale ** 2
    sigmas = noise_grid(config, points)
    log_ml, means, variances = [], [], []
    for sigma in sigmas:
        # Marginal likelihood y ~ N(0, Phi S0 Phi^T + sigma^2 I).
        cov = (phi * prior_var) @ phi.T + sigma ** 2 * np.eye(len(y))
        chol = np.linalg.cholesky(cov)
        alpha = np.linalg.solve(chol, y)
        log_ml.append(-0.5 * alpha @ alpha - np.log(np.diag(chol)).sum()
                      - 0.5 * len(y) * np.log(2 * np.pi))
        precision = phi.T @ phi / sigma ** 2 + np.diag(1 / prior_var)
        post_cov = np.linalg.inv(precision)
        post_mean = post_cov @ phi.T @ y / sigma ** 2
        means.append(phi_q @ post_mean)
        variances.append(np.einsum('qi,ij,qj->q', phi_q, post_cov, phi_q) + sigma ** 2)
    log_ml = np.asarray(log_ml)
    weights = np.exp(log_ml - log_ml.max())
    weights /= weights.sum()
    means, variances = np.asarray(means), np.asarray(variances)
    mean = weights @ means
    total = weights @ (variances + means ** 2) - mean ** 2
    aleatoric = float(weights @ sigmas ** 2)
    return {'mean': mean, 'total_var': total,
            'aleatoric_var': np.full_like(mean, aleatoric),
            'epistemic_var': total - aleatoric,
            'noise_std_posterior_mean': float(weights @ sigmas)}


def _bernoulli_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def logistic_posterior_predictive(config, x_context, y_context, x_query,
                                  slope_points=96, boundary_points=240):
    """Bayes oracle for 1-D classification episodes (entropies in nats).

    Prior: sign +-1 equally likely, |w| log-uniform on [slope_min, slope_max],
    boundary c uniform on [-boundary_range/2, boundary_range/2], and
    p(y=1 | x) = sigmoid(w (x - c)). Integrated on a grid (uniform in log|w|
    and c, so grid points have equal prior weight). As in VUD, total =
    H[E p], aleatoric = E[H[p]] and epistemic = total - aleatoric (the mutual
    information between y and the parameters).
    """
    if config.input_size != 1:
        raise ValueError('The logistic oracle grid is one-dimensional')
    x = np.asarray(x_context, dtype=float).reshape(-1)
    y = np.asarray(y_context, dtype=float)
    xq = np.asarray(x_query, dtype=float).reshape(-1)
    magnitude = np.exp(np.linspace(np.log(config.slope_min), np.log(config.slope_max), slope_points))
    slopes = np.concatenate([-magnitude, magnitude])
    half = config.boundary_range / 2
    centres = np.linspace(-half, half, boundary_points)
    w = np.repeat(slopes, boundary_points)[:, None]
    c = np.tile(centres, len(slopes))[:, None]
    logits = w * (x[None, :] - c)
    # log sigmoid(z) = -log(1 + e^-z), computed stably.
    log_lik = (y * -np.logaddexp(0, -logits) + (1 - y) * -np.logaddexp(0, logits)).sum(1)
    weights = np.exp(log_lik - log_lik.max())
    weights /= weights.sum()
    p = 1 / (1 + np.exp(-w * (xq[None, :] - c)))
    mean = weights @ p
    total = _bernoulli_entropy(mean)
    aleatoric = weights @ _bernoulli_entropy(p)
    return {'p1': mean, 'total_entropy': total, 'aleatoric_entropy': aleatoric,
            'epistemic_entropy': total - aleatoric}
