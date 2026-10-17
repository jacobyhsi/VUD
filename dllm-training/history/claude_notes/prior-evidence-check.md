---
name: prior-evidence-check
description: Screen candidate MDLM training priors with the exact-Bayes/ideal-VUD check before training; TabPFN prior was the first to fix moons_1
metadata:
  node_type: memory
  type: project
  originSessionId: ca6189b9-787a-42df-9a07-b2f20dcffb95
  modified: 2026-10-01T13:07:57.447Z
---

Before generating a bank and training an MDLM on a new prior, run `dllm-training/tmp/scripts/mode1_bag.py <dataset>` (archived in `dllm-training/history/scripts/dllm_training_tmp/` since the 2026-10-11 move; restore it and fix its paths): exact Bayes + VUD-on-exact-posterior per prior member/bag, scored against the true aleatoric (no training needed).

Findings (2026-10-01): PFN-paper GP (Bernoulli or median labels) and the paper's BNN prior all give `moons_1` near-data Va error of about 0.51, because none puts mass on clean curved boundaries. The TabPFN v1 MLP prior (non-causal branch, `small_mdlm/vendor/tabpfn_prior.py`) gives `moons_1` 0.18 near / 0.09 away and `logistic_regression` 0.27 / 0.14. Rechecked with ESS 83: 0.19 / 0.10. The trained MDLM-TabPFNClassify2D-0.8M (bank `data_banks/tabpfn_cls_3m`) got real-VUD errors of `logistic_regression` 0.083 near / 0.379 away and `moons_1` 0.217 / 0.216 (mean 0.224, best so far; Dream 0.274). It fits its prior less closely than the GP checkpoints (|Δp| 0.07–0.14).

**Why:** a training run costs about 2.5 h; the check is about 10 min and separates prior error from VUD-estimator error and MDLM error.
**How to apply:** screen any new prior or bag with this check first; extends [[pfn-prior-foundation]].
