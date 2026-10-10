"""GP-mixture classification prior from "Transformers Can Do Bayesian Inference".

Adapted from automl/TransformersCanDoBayesianInference (Apache-2.0), commit
9c20031: priors/fast_gp_mix.py (get_model, the sampling part of get_batch) and
priors/binarized_regression.py (regression_prior_to_binary). The sampling
logic and default hyperpriors are unchanged:
  x ~ U[0, 1]^d
  f ~ GP(0, outputscale * Matern_{nu=2.5}(lengthscale)) + Gaussian noise, with
      lengthscale ~ Gamma(3, 6), outputscale ~ Gamma(0.5, 0.15),
      noise ~ Gamma(1.1, 0.05), sampled by pyro_sample_from_prior
  y ~ Bernoulli(sigmoid(f))
Local changes: import fixes for newer botorch (MIN_INFERRED_NOISE_LEVEL moved;
fitting code dropped), no range rejection/min-max options, and one
hyperparameter draw per dataset (batch_size_per_gp_sample=1). Also returns
the latent f so sigmoid(f) is each label's true probability.

hyperparameters['noise'] fixes the latent noise variance instead of sampling
it, as the paper's fast_gp prior does. The GP-mixture's default noise prior
(Gamma(1.1, 0.05), median variance ~16 vs outputscale ~1.25) is meant for
regression; binarized, it makes labels nearly unpredictable (Bayes accuracy
~57%). The paper's tabular *classification* GP config uses noise 0.001.
"""
import gpytorch
import torch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior

MIN_INFERRED_NOISE_LEVEL = 1e-4  # botorch.models.gp_regression's value before it moved


def get_model(x, y, hyperparameters: dict):
    aug_batch_shape = SingleTaskGP(x, y.unsqueeze(-1))._aug_batch_shape
    noise_prior = GammaPrior(hyperparameters.get('noise_concentration', 1.1), hyperparameters.get('noise_rate', 0.05))
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=aug_batch_shape,
        noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=noise_prior_mode),
    )
    model = SingleTaskGP(x, y.unsqueeze(-1),
                         covar_module=gpytorch.kernels.ScaleKernel(
                             gpytorch.kernels.MaternKernel(
                                 nu=hyperparameters.get('nu', 2.5),
                                 ard_num_dims=x.shape[-1],
                                 batch_shape=aug_batch_shape,
                                 lengthscale_prior=GammaPrior(hyperparameters.get('lengthscale_concentration', 3.0),
                                                              hyperparameters.get('lengthscale_rate', 6.0)),
                             ),
                             batch_shape=aug_batch_shape,
                             outputscale_prior=GammaPrior(hyperparameters.get('outputscale_concentration', .5),
                                                          hyperparameters.get('outputscale_rate', 0.15)),
                         ), likelihood=likelihood)
    sampled_model = model.pyro_sample_from_prior()
    if 'noise' in hyperparameters:
        sampled_model.likelihood.noise = torch.ones_like(sampled_model.likelihood.noise) * hyperparameters['noise']
    return sampled_model, sampled_model.likelihood


@torch.no_grad()
def sample_classification_datasets(count, seq_len, num_features, hyperparameters=None):
    """count datasets of seq_len points: x (count, seq_len, d), latent f and labels y (count, seq_len)."""
    hyperparameters = hyperparameters or {}
    x = torch.rand(count, seq_len, num_features)
    latent = []
    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations', (True, True, True))):
        for i in range(count):
            model, likelihood = get_model(x[i:i + 1], torch.zeros(1, seq_len), hyperparameters)
            with gpytorch.settings.prior_mode(True):
                latent.append(likelihood(model(x[i:i + 1])).sample())
    latent = torch.cat(latent)
    return x, latent, torch.sigmoid(latent).bernoulli()
