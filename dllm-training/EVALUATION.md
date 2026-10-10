# Held-out regression evaluation

`evaluate.py` tests a fixed checkpoint against least squares and shuffled-label
controls. It never trains or updates the model. Run it inside a GPU allocation
through `screen` and `srun`, following the repository's `AGENTS.md`:

```bash
.venv/bin/python dllm-training/evaluate.py \
  --checkpoint dllm-training/outputs/linear_seed0/checkpoint.pt \
  --output dllm-training/outputs/linear_seed0/evaluation_seed2026 \
  --device cuda --tasks 256 --draws 32 --seed 2026
```

Choose a fresh output directory for each run. The evaluator saves a JSON
summary, per-task CSVs, all generated samples in compressed NumPy archives,
and `comparison.png`. The summary includes a checkpoint SHA-256 fingerprint.

## Design

- 256 independent test tasks per suite, separate from training/validation.
- D=15 observed demonstrations and one fully hidden query output.
- 32 stochastic MDLM samples per task, 32 diffusion steps, temperature 1.
- Correct contexts and contexts with observed labels randomly permuted across
  covariates. Query covariates and evaluation targets stay unchanged. No latent
  weights, bias, or query labels enter the model prompt.
- Matching sample seeds and batch layouts across the two conditions reduce
  sampling noise in their difference. They do not make the two conditions
  generate identical numerical samples.
- Fitted least squares includes an intercept and receives the same rounded
  covariates and observed labels as the tokenizer. We also report shuffled
  least squares, context-mean, zero, and ground-truth-function baselines.

The **training-family** suite uses the checkpoint's synthetic linear task
distribution, restricted to D=15 and one query. The **VUD-style linear** suite
uses slope -1, intercept 3, observation-noise standard deviation 2, context
covariates from N(1, 2^2), and a uniform query in [-15, 15). These parameters
match `datasets_toy_regression/linear_regression/info.json`. Unlike the VUD
runner, this diagnostic resamples a new context for every query, uses a random
query rather than a grid, and rounds at the tokenizer's two-decimal precision
rather than pre-rounding the dataset to one decimal place. It is therefore a
distribution-level diagnostic, not a reproduction of an existing VUD run.

## Reading the scores

The primary score is MSE of each method's **predicted mean** against the true
noiseless function at the query. For MDLM this mean is estimated from 32
samples, so Monte Carlo error remains. Scores against independently noisy
query targets are also saved, as are mean predictive variance and average
single-draw error. Do not directly compare single-draw error with deterministic
least-squares error as if they measured the same quantity.

The shuffle effect is `MSE(shuffled) - MSE(correct)` on matched tasks. Positive
values indicate worse predictions after shuffling. Its 95% interval uses a
paired bootstrap over tasks (2,000 replicates); the other MSE intervals also
bootstrap tasks, not individual generated samples. An interval spanning zero
does not prove complete context independence: it means this test did not find
a reliable predictive benefit from the correct label association.

This is not yet a joint-distribution, uncertainty-calibration, or VUD
decomposition evaluation. Establish useful regression prediction first.
