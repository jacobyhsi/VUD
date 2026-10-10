"""Prior bag over PFN classification label priors.

Follows automl/PFNs `pfns/priors/prior_bag.py` (Apache-2.0; read at commit
f04d3ca, SamuelGabriel/PFNs mirror): each dataset is drawn from one member
prior, chosen with probability softmax(weights). PFNs passes the first
member's weight as 1.0 and the others as `prior_bag_exp_weights_i`; here every
member's logit is given explicitly. Local differences: assignment per dataset
(PFNs assigns per group of `batch_size_per_gp_sample` datasets) and drawn with
the caller's numpy generator, so banks stay reproducible.

Members, all from the PFN lineage. Two are label mechanisms applied to one
latent GP sample f (see pfn_priors.py):
  'median'    y = 1[f > median of f over the dataset]  (paper, Section 6);
              deterministic labels, so the latent is stored saturated (+-30)
              and sigmoid(latent) is the true (0/1) label probability
  'bernoulli' y ~ Bernoulli(sigmoid(f))  (released code, binarized_regression.py)
and two are function priors that bring their own inputs and labels:
  'bnn'       the paper's tabular BNN prior (pfn_mlp_prior.py)
  'tabpfn'    TabPFN v1's MLP prior, non-causal branch (tabpfn_prior.py)
  'tabpfn_causal'  TabPFN v1's MLP prior, causal branch (tabpfn_prior.py); latents are the
              saturated observed labels, since p(y | x) has no closed form there
  'tabpfn_gp' TabPFN v1's GP member (tabpfn_prior.py): noisy threshold labels,
              including very short lengthscales and weak-signal tasks
  'controlled_gp' Local smooth/noise-controlled GP adaptation (controlled_gp_prior.py).
              Explicit variance ratios and a task-level independent threshold;
              this is NOT the released TabPFN prior.
CODES numbers the members in a bank's labelling.npy.
"""
import numpy as np

MEMBERS = ('median', 'bernoulli', 'bnn', 'tabpfn', 'tabpfn_causal', 'tabpfn_gp', 'controlled_gp')  # append only: preserve legacy assignments
GP_MEMBERS = ('median', 'bernoulli')
CODES = {'bernoulli': 0, 'median': 1, 'bnn': 2, 'tabpfn': 3, 'tabpfn_causal': 4, 'tabpfn_gp': 5, 'controlled_gp': 6}


def parse_weights(spec):
    """'median=0,bernoulli=0' -> {'median': 0.0, 'bernoulli': 0.0} (logits)."""
    weights = {}
    for item in filter(None, spec.split(',')):
        name, value = item.split('=')
        if name not in MEMBERS:
            raise ValueError(f'Unknown prior-bag member {name!r}; choose from {MEMBERS}')
        weights[name] = float(value)
    if not weights:
        raise ValueError('Prior bag needs at least one member')
    return weights


def member_probabilities(weights):
    names = [name for name in MEMBERS if name in weights]
    logits = np.array([weights[name] for name in names])
    p = np.exp(logits - logits.max())
    return names, p / p.sum()


def assign(rng, count, weights):
    """Member name per dataset; one uniform draw per dataset whatever the bag holds."""
    names, p = member_probabilities(weights)
    index = np.searchsorted(np.cumsum(p), rng.random(count), side='right')
    return np.array(names)[np.minimum(index, len(names) - 1)]


def apply_labels(members, latent, y):
    """Relabel datasets assigned to 'median' in place; 'bernoulli' keeps the sampled labels."""
    median = members == 'median'
    if median.any():
        above = latent[median] > np.median(latent[median], axis=1, keepdims=True)
        y[median] = above
        latent[median] = np.where(above, 30.0, -30.0)
    return median
