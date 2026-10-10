"""BNN (random-MLP) classification prior from "Transformers Can Do Bayesian Inference".

Adapted from automl/TransformersCanDoBayesianInference (Apache-2.0), commit
9c20031: priors/mlp.py (get_batch, non-causal path) with the paper's tabular
BNN configuration (notebooks/TabularEvalSimple.ipynb, bnn_model_config; read
by tabular.get_mlp_prior_hyperparameters):
  x ~ N(0, 1)^d
  3 layers: Linear(d, h) -> [Tanh, Linear(h, h), +noise] -> [Tanh, Linear(h, 1), +noise]
  h = scaled_beta(2.0, 4.0, 150, 2), every weight and bias ~ N(0, sigma^2),
  sigma ~ Gamma(k=3.6188, theta=0.06774), noise std ~ Gamma(k=1.8663, theta=0.05275),
  dropout 0
  y = 1[out > median of out over the dataset]   (priors/utils.py, Binarize)
Local changes: networks are batched (units beyond h are masked, which leaves
the function unchanged); inputs are left unnormalized (episodes z-score them by
the context); order_y and the categorical/used-feature options are dropped
(our datasets are shuffled and use every feature). Also returns each point's
true label probability P(out(x) > t), with t the dataset's realized median and
the hidden noise integrated out by Monte Carlo.
"""
import torch

LAYERS_HIDDEN = (2.0, 4.0, 150, 2)        # scaled_beta_sampler_f(a, b, scale, minimum)
SIGMA_GAMMA = (3.6187797729244253, 0.06773738681062867)
NOISE_GAMMA = (1.8663049257557085, 0.05275478076173361)


def _gamma(shape, k, theta, generator, device):
    return torch._standard_gamma(torch.full(shape, k, device=device), generator=generator) * theta


def sample_networks(count, num_features, generator=None, device='cpu'):
    """count random networks: dict of batched weights, sigma and noise std."""
    a, b, scale, minimum = LAYERS_HIDDEN
    # scaled_beta_sampler_f: minimum + round(Beta(a, b) * (scale - minimum + 1) - 0.5)
    u = torch.rand(count, device=device, generator=generator)
    hidden = minimum + torch.round(_beta_icdf(a, b, u) * (scale - minimum + 1) - 0.5)
    sigma = _gamma((count,), *SIGMA_GAMMA, generator, device)
    noise = _gamma((count,), *NOISE_GAMMA, generator, device)
    mask = (torch.arange(scale, device=device)[None] < hidden[:, None]).float()   # (count, H)
    n = lambda *s: torch.randn(count, *s, device=device, generator=generator) * sigma.view(-1, *[1] * len(s))
    return {'w1': n(num_features, scale) * mask[:, None], 'b1': n(scale) * mask,
            'w2': n(scale, scale) * mask[:, None] * mask[:, :, None], 'b2': n(scale) * mask,
            'w3': n(scale, 1) * mask[:, :, None], 'b3': n(1),
            'noise': noise, 'hidden': hidden, 'sigma': sigma}


def _beta_icdf(a, b, u):
    """Beta(a, b) quantiles from a tabulated CDF (torch has no Beta icdf), so the given generator drives sampling."""
    grid = torch.linspace(0, 1, 2049, device=u.device)[1:-1]
    cdf = torch.cumsum(grid ** (a - 1) * (1 - grid) ** (b - 1), 0)
    cdf = cdf / cdf[-1]
    return grid[torch.searchsorted(cdf, u).clamp(max=len(grid) - 1)]


def forward(net, x, generator=None):
    """x (count, N, d) -> noisy outputs (count, N); fresh hidden noise for every point."""
    std = net['noise'].view(-1, 1, 1)
    a = x @ net['w1'] + net['b1'][:, None]
    a = torch.tanh(a) @ net['w2'] + net['b2'][:, None]
    a = a + std * torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)
    out = torch.tanh(a) @ net['w3'] + net['b3'][:, None]
    out = out + std * torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
    return out.squeeze(-1)


def median_threshold(out):
    """np.median over the dataset (mean of the two middle values for even counts)."""
    s, n = out.sort(-1).values, out.shape[-1]
    return (s[..., (n - 1) // 2] + s[..., n // 2]) / 2


def label_probs(net, x, t, draws=64, generator=None):
    """P(out(x) > t) per point, integrating the hidden noise with `draws` passes."""
    hits = sum((forward(net, x, generator) > t[:, None]).float() for _ in range(draws))
    return hits / draws


@torch.no_grad()
def sample_classification_datasets(count, seq_len, num_features, draws=64, generator=None, device='cpu'):
    """x (count, seq_len, d), latent = logit of the true P(y=1), and labels y (count, seq_len)."""
    net = sample_networks(count, num_features, generator, device)
    x = torch.randn(count, seq_len, num_features, device=device, generator=generator)
    out = forward(net, x, generator)
    t = median_threshold(out)
    y = (out > t[:, None]).float()
    # Clipped at half a draw, so the logit stays finite where every pass agrees.
    p = label_probs(net, x, t, draws, generator).clamp(0.5 / draws, 1 - 0.5 / draws)
    return x, torch.log(p / (1 - p)), y
