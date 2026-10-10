# Controlled TabPFN episode-geometry ablation

Purpose: test whether nearby-example conditioning improves MDLM predictions
used by VUD. The VUD estimator, model, tokenizer, normalization, optimizer,
and TabPFN network/noise hyperpriors are unchanged.

## Paired design

- Same sampled TabPFN network and noise scales in each pair; 30% 1D and 70% 2D.
- Same ordinary context observations, context size (3–32), hidden-query count
  (one or two, equally likely), and task split. The labelled auxiliary is
  included in the observed count in single-query episodes.
- **IID control:** both auxiliary and final query follow the sampled network's
  Gaussian covariate distribution.
- **Geometry intervention:** with probability .5, sample the final query
  uniformly within ±6 context standard deviations per feature; otherwise use
  the control query. With probability .7, place the auxiliary at the query plus
  Gaussian perturbation with standard deviation .1 times context feature std.
  Otherwise use the control auxiliary. These are explicit starting settings,
  not tuned optima. The .1 perturbation mirrors the current VUD configuration.
- Every new label is evaluated at its own input using the same latent network
  and fresh row-wise observation noise. Labels are not copied from neighbours.
- The nearby auxiliary is observed in single-query episodes and hidden in
  two-query episodes. Observed rows and hidden auxiliary positions are shuffled.
- Baseline backend normalization is retained, including recomputation after
  adding an observed auxiliary. This is NOT a normalization ablation.

## Threshold caveat

The released-prior adaptation derives its rank/value threshold from a finite
dataset. We calibrate it on 34 IID points, then hold it fixed when evaluating
new inputs. Network, noise and threshold-calibration rules are unchanged, but
the original query/threshold finite-sample coupling is not. The paired IID
control applies exactly the same extension, so comparisons to it isolate
episode geometry. Do not attribute a difference versus the old checkpoint
solely to query placement.

## Training and evaluation

Each arm has 3,006,464 episodes, with the last 20,000 reserved (10,000 validation,
10,000 test). Paired task IDs are split identically to prevent cross-arm leakage.
Only episode-specific layouts are loaded; padding is excluded. Legacy PFN
banks retain their original random-subset loader.

Both models start from seed-0 random initialization: 4 blocks, width 128,
4 heads, 799,766 parameters; 20,000 updates, batch 256, LR .0003, FP32. Evaluated
checkpoints are retained every 2,000 updates. This matches the original small
TabPFN training budget, not the longer/larger follow-ups.

`evaluate_tabpfn_conditionals.py` scores held-out query labels before and after
observing the auxiliary, using the actual backend preprocessing. It also
reports marginal-consistency error and response to switching the auxiliary
label. These are proper predictive scores / consistency diagnostics, not an
exact-Bayes comparison or a ground-truth epistemic estimate. Larger conditional
response alone is not a success criterion.

Select each arm's checkpoint by conditional log-loss on the SAME 1,024 held-out
geometry-validation episodes. Then evaluate on disjoint IID and geometry test
episodes, the existing 40-context predictive evaluation (logistic/moons1/moons2),
and vanilla VUD grids for logistic and moons1. Final VUD contexts are not used
for checkpoint selection. Multi-seed VUD evaluation remains a follow-up.

Launcher: `tmp/scripts/run_tabpfn_geometry_ablation.sh` from the repository root,
inside screen and an existing Slurm allocation. It uses GPU0; the existing
Dream spirals run on GPU1 is not changed. Bank generation precedes the two
training processes. Output and log names are distinct from all older runs.

Tests: `test_tabpfn_episodes.py` covers exact equality with geometry disabled,
paired contexts, label correctness under a known latent function, close-query
geometry, deterministic generation, disjoint splits, padding, and target
isolation. The temporary smoke launcher additionally tests CUDA bank generation,
training, resume, retained snapshots and conditional checkpoint selection;
it cleans its temporary artifacts on exit.
