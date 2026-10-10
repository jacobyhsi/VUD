# Controlled GP classification experiment

This is a **local prior adaptation**, not a reproduction of the released PFN or
TabPFN prior. The existing `tabpfn_gp` generator and checkpoints are unchanged.
No moons/logistic evaluation labels are used to generate tasks or select models.

## Why this experiment

The released-checkpoint TabPFN GP hyperpriors put substantial mass on extremely
short lengthscales and on observation noise larger than signal variance. Adding
more of those tasks does not specifically add smooth, nearly clean boundaries.
We test explicit lengthscale and noise/signal settings before changing model size.

## New task distribution

`small_mdlm/vendor/controlled_gp_prior.py` and `configs/controlled_gp_prior.json`:

- Inputs: IID standard Gaussian, in one or two dimensions (30% 1D as before).
- Function: one zero-mean RBF GP per dataset, signal variance one, lengthscale
  log-uniform on [0.5, 3] in the **raw standard-Gaussian input coordinates**.
- Noise: each dataset is clean-regime or noisy-regime with probability 0.5.
  Observation-noise/signal **variance** ratio is log-uniform on [1e-6, 1e-3]
  or [0.03, 1], respectively. Noise is independent between observations.
- Threshold: one independent N(0, 0.5^2) draw per task; labels are
  `1[f(x) + epsilon > threshold]`. This changes TabPFN's rank/value threshold
  mechanism as well as its lengthscale/noise hyperpriors. Independence makes
  the task's predictive law well-defined at additional query locations.
- Stored probabilities are Phi((f(x)-threshold)/sqrt(noise_variance)), clipped
  only for storing finite logits. These are probabilities given the sampled
  function, **not** Bayes posterior predictions given context.
- Numerical covariance jitter 1e-8; diagonal normalized back to signal variance one.

These ranges and weights are starting hypotheses, not paper defaults or tuned
optima. A clean function can still have an uncertain decision boundary near its
threshold, and clean observation noise does not imply low posterior uncertainty.

## Matched training profile

50% original non-causal TabPFN MLP + 50% controlled GP. Within the GP half,
clean/noisy regimes are balanced. This matches the old GPMLP50 mixture weight.
Use `configs/tabpfn_controlled_gp50_classify2d.json`: 4 blocks, width 128,
4 heads, numeric row embeddings; 20k steps, batch 256, LR .0003, FP32, seed 0.
The original random-subset episode loader, context-only normalization,
3--32 demonstrations, and one/two-query objective are retained. GP lengthscales
are not specified after normalization, since context std varies by episode.

The bank is new; old banks, outputs, plots, and active runs are never replaced.
Bank metadata records the complete controlled configuration and member weights.
This is a combined prior intervention, not an isolated lengthscale ablation.

## Checks and automated evaluation

Before training: unit tests, CUDA bank/train/resume smoke check, generated-task
plots and entropy/lengthscale/noise summaries. A Monte Carlo posterior screen
uses the controlled GP's own coordinates and likelihood, reports effective
sample size, and never certifies MDLM's normalized-input posterior. Low ESS
marks a reference inconclusive rather than evidence of good/bad model learning.

After training: proper held-out conditional label scores before/after observing
an auxiliary, disjoint validation/test splits, 40-context predictions on the
existing logistic/moons datasets, and unchanged vanilla VUD grids on
`logistic_regression` and `moons_1`, context seed 0. All intermediate checkpoints
are retained, but final 20k weights are used for VUD to match earlier runs;
evaluation labels are not used for checkpoint selection.

Launch inside screen + the existing Slurm allocation with
`tmp/scripts/run_controlled_gp_ablation.sh`. It uses the GPU chosen after a
read-only VRAM check and leaves existing Dream runs alone. All runtime temporary
files and logs are in the repository's `tmp/` directory.
