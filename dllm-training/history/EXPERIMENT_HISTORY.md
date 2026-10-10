# Experiment history: small MDLMs for VUD

Read this first when picking the project up again. It records everything tried so far, what worked and
what didn't (with numbers), the decisions taken, and where the evidence lives. Last updated 2026-10-10.

Companion material in this folder:
- `claude_notes/`: working rules and lessons kept by the coding assistant.
- `scripts/`: the scratch launchers and analysis scripts behind every experiment, kept for reference.
- `checkpoints/`: the key trained models.
- `transcripts/`: the full assistant conversation logs, redacted.

Raw results live in `results/` at the repository root.

## 1. Goal and conventions

**Goal:** train small masked-diffusion LMs (an MDLM: a DiT backbone with masked diffusion; code in
`dllm-training/`) that do in-context learning, and use them as VUD models alongside Dream-v0-Base-7B.
VUD (variational uncertainty decomposition) splits predictive uncertainty into aleatoric and epistemic
parts using only an in-context learner:

| Quantity | Definition |
|---|---|
| **TU** | H[p(y \| x, D)] |
| **Va (≈ AU)** | E_u H[p(y \| x, z, u, D)], minimised over auxiliary inputs z: take the min Va among the 5 z with the lowest KL(p(y\|x,D) ‖ p(y\|x,z,D)), capped at TU |
| **Ve (≈ EU)** | TU − Va |

**Units:**
- Classification is measured in **bits**: 15 z ~ N(x, (0.1 · context std)²), rounded to 1 decimal.
- Regression is measured in **nats**: num_z 5 and 12 samples per Gaussian (IQR std).

**Toy datasets:**
- Classification: logistic_regression (1-D, x range ±15, D 15), moons_1, moons_2 (D 30), spirals.
- Regression: linear_regression (±15, D 15), gaps (D 30, `--x_range "{'x': [-15, 15, 0.2]}"`).

**Scoring used throughout:**
- **AU error** = |Va − true aleatoric|. ✅ ≤ 0.25.
- **Far-field EU:** moons ✅ ≥ 0.45, 🟡 0.25–0.45; LR ✅ ≥ 0.16, 🟡 0.12–0.16.
- **Near / far:** within the context range (1-D) or within 0.35 of a context point (2-D).

## 2. Final decisions

| Task | Adopted model | Checkpoint | Results name |
|---|---|---|---|
| **Classification** (adopted 2026-10-07) | 0.8M DiT. Prior: 50% controlled GP (`small_mdlm/vendor/controlled_gp_prior.py`, `configs/controlled_gp_prior.json`) + 50% TabPFN v1 MLP non-causal (our port, `small_mdlm/vendor/tabpfn_prior.py`). Paired geometry episodes, 1-D share 0.5, structured input layouts 75%, 20k steps | `outputs/tabpfn_controlled_gp50_geometry_1d_layout75/checkpoint_step_020000.pt` (copy in `checkpoints/`) | `MDLM-TabPFN-ControlledGP50-Geometry1D-Layout75-0.8M` |
| **Regression** (candidate; adoption recommended, not yet confirmed) | 0.8M DiT. Prior: 50% heteroskedastic GP (lengthscale 0.1–3) + 50% TabPFN v1 MLP in regression mode; geometry episodes, 75% layouts, x and y z-scored by the context, 40k steps (step 30k selected) | `outputs/regression_hetgp50_geometry/checkpoint_step_030000.pt` (copy in `checkpoints/`) | `MDLM-Regress-HetGP50-Geometry-0.8M` |

**Final classification results** (bits; AU error near / far; EU far):

| | AU error near / far | EU far | TU far | Accuracy |
|---|---|---|---|---|
| logistic_regression | 0.16 / 0.22 | 0.25 | — | — |
| moons_1 | 0.21 / 0.23 | 0.49 | 0.73 | — |
| moons_2 | 0.21 / 0.25 | 0.50 | 0.75 | 0.786 |

**Predictive accuracy / log-loss** (40 contexts):

| Model | LR | moons_1 | moons_2 |
|---|---|---|---|
| TabPFN-MLP MDLM (B6) | 0.553 / 0.752 | 0.930 / 0.192 | 0.778 / 0.553 |
| **Final** | 0.588 / 0.737 | 0.964 / 0.101 | 0.786 / 0.524 |
| BayesClassify 2.5M | 0.598 / 0.730 | 0.980 / 0.086 | 0.793 / 0.479 |
| LR baseline | 0.612 / 0.728 | 0.843 / 0.374 | 0.784 / 0.458 |
| GP classifier | 0.572 / 0.707 | 0.985 / 0.258 | 0.781 / 0.492 |
| true p | 0.670 / 0.613 | 1.0 / 0.001 | 0.829 / 0.350 |

**Regression candidate** (nats):

| | AU | EU |
|---|---|---|
| linear_regression | 1.69 on the data; 2.06 far (best of all) | 0.72 beyond (previous 1.14) |
| gaps | Tight clusters +0.01 (previous best MDLM +0.53, Dream +0.79, exact −0.20, true −0.88). Noisy cluster 1.34 (true 2.11, Dream 1.72) | 0.70 in the gaps (previous 0.27, Dream 0.22); 0.97 beyond |

- **Status:** LR TU ✅ AU 🟡 EU 🟡; gaps TU ✅ AU 🟡 EU ✅.
- **Calibrated** (even under-confident far away) on its own prior's tasks.
- **Open:** adopt it, or run one more round for the gaps noisy cluster.
- **Agreed not to chase LR far-field EU:** the exact LR far-field target is unstable (EU 3.70 at ESS 21 against 1.61 at ESS 383), driven by wildly extrapolating MLP networks.

## 3. Lineage (what was tried, in order)

**Status legend:** ✅ working · 🟡 partly · ❌ not working
**Change vs the step it built on:** ⬆️ better · ⬇️ worse · ↔️ mixed or same

Config files are in `configs/`; results names are in `results/toy_*/<dataset>/<name>/`.

### Phase A: making a small diffusion LM do in-context learning
- **A1. Digit-token MDLM:** numbers as digit tokens. ❌ It couldn't learn: digits alone don't carry sign and value jointly. (`digits_*` configs)
- **A2. Row-vector value embedding** ⬆️: each token also gets its row's numbers (x, y, hidden flag). ✅ Every later model uses this. (`row_digits_*`, `value_digits_*`)
- **A3. Small fp32 model** ⬆️: 4 blocks × 128 width (≈ 0.8M parameters), no timestep conditioning. bf16 had stalled. ✅ (`ablate_linear_*`)
- **A4. MDLM-BayesLinear-1.3M** (Bayesian linear-regression prior), on LR: TU ✅, AU 🟡 (1.67 against the true 2.11), EU ✅.
- **A5. MDLM-BayesRegress / BayesRegressGP-0.8M** ↔️ (broader hand-built regression priors):
  - LR: AU 1.42–1.52, EU 1.14–1.29 beyond.
  - gaps: AU ❌ (tight clusters too noisy); EU ❌ in the gaps, ✅ beyond.
  - **Concern raised:** the priors were being tailored to the test datasets.
- **A6. MDLM-BayesClassify2D-2.5M / 0.8M** (hand-built classification families): LR and moons_1 TU ✅, AU 🟡, EU ✅. It was designed after seeing the datasets. **Decision:** rebuild on the PFN paper's priors.

### Phase B: classification on PFN-lineage priors
- **B1. PFN-paper GP prior** (latent noise 0.001; the default made labels coin flips): LR AU 0.06 inside, 0.49 far; EU ❌. (`pfn_classify2d_noise1e-3`)
- **B2. BoTorch outputscale** ↔️: moons_1 TU, AU and EU all ❌. The model thinks the moons are noisy. (`_os2`)
- **B3. Mode-1 check:** the MDLM matched its prior's exact Bayes answer, so **the prior was the problem**.
- **B4. Mixed labels** (half clean median-threshold, half noisy) ↔️: LR EU ⬆️, AU ⬇️; moons_1 still ❌. The exact check showed the gain came from the model *drifting* from its prior. (`_os2_mixed`)
- **B5. Exact-Bayes screening of priors before training:** the paper's BNN prior ❌; **the TabPFN v1 MLP prior ✅ fixed moons_1**.
- **B6. TabPFN v1 MLP prior** ⬆️ (`tabpfn_classify2d`, MDLM-TabPFNClassify2D-0.8M): moons_1 AU ✅ fixed, but TU ❌ (confidently wrong far away), EU ❌.
- **B7. 60k steps / 2.7M parameters** ⬇️: aleatoric haze on moons_1. (`_long`, `_big`)
- **B8. Geometry episodes on the MLP prior** ↔️: a labelled point next to the query plus far queries. moons_1 TU ❌ (overconfident).
- **B9. TabPFN causal + non-causal MLP** ↔️: LR AU ✅, moons_1 ⬇️ (noisy again). (`tabpfn_full_classify2d`)
- **B10. TabPFN GP + MLP (50/50, 4/96)** ⬇️: TabPFN's own GP hyperprior is extreme, so it calls everything noise far out. (`tabpfn_gpmlp*`)
- **B11. Controlled GP + MLP 50/50** ↔️ (our smoother GP: lengthscale 0.5–3, 50% clean / 50% noisy, data-independent threshold): best LR AU inside; moons_1 haze far away.
- **B12. Exact conditional reference** p(y | x, z, u): the prior's far-field EU is ≈ 0.53, but the model under-reacted to z's label. **A training problem.**
- **B13. Controlled GP50 + geometry** ⬆️, with an IID control: moons_1 haze gone, EU 0.22 → 0.37; moons_2 EU 0.18 → 0.38. The IID control did not improve, so **geometry was the cause**.
- **B14. 1-D share 30% → 50%** ⬆️ (the sibling "more geometry" was ⬇️): LR EU ✅ (matches its prior); moons_1 AU best of all, EU 0.37.
- **B15. 75% GP / 2.7M parameters** ⬇️.
- **B16. Calibration diagnostic:** calibrated on its own prior's tasks. moons_1's structured two-arc input layout was the cause of its overconfidence.
- **B17. + structured input layouts (50%)** ⬆️ (clusters, curves, correlated Gaussians): moons_1 TU 0.57 → 0.69, EU 0.37 → 0.45.
- **B18. Round 5:** 40k steps (arm A) vs **75% layouts (arm B, adopted as final)** ⬆️. moons_1 TU 0.73, EU 0.49; moons_2 all ✅.

**Path:** A6 → B1 → B2 ↔️ → B4 ↔️ → B6 ⬆️ → (B7–B11 sideways or worse) → B13 ⬆️ → B14 ⬆️ → B15 ⬇️ → B17 ⬆️ → **B18 (final)**

### Phase C: regression on the same recipe
- **C1. Measurement check:** VUD's 12-sample estimator can resolve σ = 0.1 (a perfect sampler gives −1.05 against a true −0.88). The old models predicted about 5× too wide.
- **C2. Exact prior screen** (`tmp/scripts/regression_reference.py`): heteroskedastic GP + TabPFN MLP regression. Broadening the lengthscale from 0.5–3 to 0.1–3 raised gaps EU from 0.27 to 0.43 (2.13 beyond). LR passes. The MLP member is useless on gaps but carries LR.
- **C3. Training round 1** ⬆️ → the regression candidate (section 2).
- **C4. Diagnosis:** calibrated; the LR exact target is unstable, so we're not chasing it.

## 4. Ground-truth and TabPFN v1 studies (2026-10-08 to 10-09)

### 4a. Simulated ground truth, our TabPFN-prior port (the "B6 vs exact Bayes" check)
**Setup:** 100 fresh tasks from our port of TabPFN's non-causal MLP prior, 2 near and 2 far queries each; exact Bayes by importance sampling. Script: `scripts/dllm_training_tmp/sim_groundtruth_classification.py`.

| (bits) | TU near / far | AU near / far | EU near / far |
|---|---|---|---|
| Reference (truth / exact Bayes) | 0.52 / 0.62 | 0.30 / 0.26 | 0.22 / 0.36 |
| VUD on exact Bayes | 0.52 / 0.62 | 0.41 / 0.38 | 0.11 / 0.24 |
| B6 + VUD | 0.50 / 0.64 | 0.45 / 0.49 | 0.05 / 0.15 |

**What it shows:**
- **B6 learns the Bayesian predictive:** TU and log-loss match Bayes, and |p − p_Bayes| ≈ 0.06.
- **VUD alone causes much of the AU/EU bias** (AU +0.11, EU only 1/2–2/3 recovered).
- **The MDLM adds a smaller bias in the same direction.**
- **Correlations:** Va vs true AU 0.58–0.66; Ve vs Bayes EU 0.59–0.76.

### 4b. TabPFN v1's ORIGINAL prior code (`generate_tabpfn_v1_prior_data.py`; see `TABPFN_V1.md`)
**Prior bag in the notebook recipe:** GP with weight 1 vs MLP/SCM with weight w ~ U(2, 10) → **≈ 4% GP**, 48% non-causal MLP, 48% SCM.
- The **released checkpoint** was trained with w ~ U(1e6, 1e6+1), i.e. **MLP/SCM only, no GP**, plus up to 100 features, 10 classes and ~1000 points.
- The SCM lives in `tabpfn/priors/mlp.py`: x and y are random hidden units of a random sparse network; `y_is_effect`, `in_clique` and `sort_features` control which.

**MDLM trained on 2.85M notebook-prior datasets** (`tabpfn_original_classify2d`), against the held-out labels of 969 fresh datasets:

| Distance from the context | All | < 1 std | 1–2 std | ≥ 2 std |
|---|---|---|---|---|
| MDLM log-loss / accuracy / TU | 0.668 / 0.772 / 0.626 | 0.657 / 0.777 / 0.615 | 0.667 / 0.773 / 0.631 | 0.722 / 0.749 / 0.658 |
| TabPFN v1 log-loss / accuracy / TU | 0.695 / 0.776 / 0.518 | 0.684 / 0.780 / 0.508 | 0.697 / 0.778 / 0.513 | 0.738 / 0.753 / 0.581 |

**What it shows:**
- **The MDLM is better calibrated:** ECE 0.016 against TabPFN's 0.054. Its TU is within 0.04 of its log-loss, against 0.18 for TabPFN.
- **TabPFN's inference deliberately sharpens predictions:** logits / 0.8, hard-coded in `transformer_predict` when there's no style. Undoing it gives TabPFN log-loss 0.667, equal to the MDLM.

### 4c. Ground-truth GP decomposition figure (4 one-feature GP datasets from TabPFN's GP path)
**Setup:** real hyperparameter draws recorded by a hook (`scripts/dllm_training_tmp/tabpfn_gp_hyper_hook.py`); oracle GP posterior with the true hyperparameters, via elliptical slice sampling. Script: `gp_groundtruth_decomposition.py`.

**What it shows:**
- **TabPFN's GP hyperprior is degenerate:** 12 of 19 one-feature draws have lengthscale < 0.01, i.e. white noise, so the label is independent of x.
- **The oracle decomposition behaves as theory says:** AU ≈ 0 for noise-free draws; AU tracks the true noise *level* where labels are noisy; EU grows with distance in lengthscales.
- **MDLM + VUD gets TU roughly right but puts most of it into AU,** even where the oracle says it is all epistemic. Part of this is unavoidable: the oracle knows the noise level, while the model must infer it from 20 points.

### 4d. The professor's three settings (classification; VUD on logistic_regression and moons_1)
**The three models:**
1. **MDLM (notebook prior):** `outputs/tabpfn_original_classify2d`, results `MDLM-TabPFNNotebook-0.8M`.
2. **MDLM (TabPFN-generated):** labels sampled from TabPFN v1's model; `outputs/tabpfn_model_seeded_classify2d`, results `MDLM-TabPFNModel-0.8M`.
   - **Seeded with 2–10 notebook labels holding both classes.** Unseeded sequential sampling **collapsed**: TabPFN gives an absent class p ≈ 0, so 72% of datasets came out single-class.
3. **TabPFN v1** (released checkpoint, as published) + VUD: `src/tabpfn_backend.py`, results `TabPFN-v1`.

| (bits) | LR AU err near / far | LR EU near / far | moons_1 AU err near / far | moons_1 EU near / far |
|---|---|---|---|---|
| MDLM (notebook prior) | 0.06 / 0.35 | 0.02 / 0.06 | 0.32 / 0.20 | 0.04 / 0.07 |
| MDLM (TabPFN-generated) | 0.20 / 0.24 | 0.05 / 0.15 | 0.23 / 0.26 | 0.06 / 0.19 |
| TabPFN v1 | 0.29 / 0.23 | 0.07 / 0.38 | 0.14 / 0.13 | 0.19 / 0.40 |
| **Final classification checkpoint** | 0.16 / 0.22 | 0.03 / 0.25 | 0.21 / 0.23 | 0.08 / 0.49 |
| Dream-7B | 0.14 / 0.28 | 0.04 / 0.16 | 0.39 / 0.28 | 0.25 / 0.47 |

**Lessons:**
1. **VUD works on a strong PFN.** TabPFN puts far-field uncertainty into EU, and its far LR AU equals the truth.
2. **The notebook-prior MDLM is blind to distance;** its TU barely changes.
3. **The training data is the lever.** TabPFN-generated data recovers the TU shape and triples far EU.
4. **What's left is MDLM-specific:** a weak response to the auxiliary point (z, u). It gets about half of TabPFN's far EU.
5. **TabPFN is over-confident at the data edges** on LR.
6. **The final checkpoint is roughly on par with TabPFN v1 + VUD,** so the project remains a go.

### 4e. Qwen2.5-14B baselines (original repo setup)
**Setup:** vLLM with FlashInfer, VUD vanilla on LR, moons_1, moons_2, linear_regression and gaps (spirals excluded). All complete; results under `results/toy_*/<dataset>/Qwen2.5-14B/`.

## 5. What didn't work (quick reference)

| Attempt | Why it failed |
|---|---|
| Digit-only tokens | No numeric value signal |
| bf16 | Training stalled |
| Hand-built priors | Tailored to the test sets |
| PFN-paper GP prior (default noise) | Labels became coin flips |
| BoTorch outputscale | moons looked noisy |
| Mixed-label bag | The gain was drift from the prior |
| PFN BNN prior | Failed the exact screen |
| Longer training / a bigger model | Aleatoric haze |
| TabPFN's own GP member, 75% GP mixes | Extreme hyperpriors |
| Causal MLP branch | moons noisy again |
| Unseeded TabPFN-model label sampling | Collapse to single-class datasets |
| Chasing LR far-field exact EU | The target is unstable |

## 6. Operational lessons (cluster and tooling)

**Running jobs:**
- Use a screen session; `srun --jobid=<job> --overlap ...`; kill the screen when done.
- Check `nvidia-smi` before every launch and balance jobs across GPUs.
- Queue follow-up steps (checks, evals, VUD) in the launcher script so long jobs continue unattended.
- **Never edit a bash script while it is running.** Write a new script, or use a watcher that kills unwanted sub-runs.

**Shell gotchas:**
- `pkill -f <pattern>` kills your own shell when the pattern appears in its command line. Match with `ps -eo pid,args | awk '$2 ~ /python$/ && /pattern/'` instead.
- `env VAR=... cmd` through srun failed because PATH contains an `env` directory. Use `bash -c 'VAR=... cmd'`.
- `rm -rf $VAR/*` is blocked by the safety check. Use literal paths.

**TabPFN v1:**
- Under SLURM with several GPUs, TabPFN's `train()` tries to start distributed training. Unset `SLURM_PROCID` / `LOCAL_RANK` and hide the GPUs for prior sampling.
- Its prior split needs `max_eval_pos` < the sequence length, or every dataset fails the class-compatibility check.
- Its code imports names from `torch.nn.modules.transformer` that current PyTorch dropped. Use the shim in `small_mdlm/tabpfn_v1.py`.

**Other:**
- Seed variation is small: don't spend runs on extra `icl_sample_seed` values.
- Screen priors with exact Bayes / ideal VUD before training. This is what found the TabPFN prior.
- Temporary files go in `tmp/` (repo) or `dllm-training/tmp/` (training work); both are git-ignored.
- `gaps` needs `--x_range "{'x': [-15, 15, 0.2]}"`.
