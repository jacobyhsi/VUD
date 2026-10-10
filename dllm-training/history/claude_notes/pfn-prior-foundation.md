---
name: pfn-prior-foundation
description: "dllm-training data generation builds on the PFN paper's priors; extend them, don't compare against hand-built priors"
metadata:
  node_type: memory
  type: project
  originSessionId: ca6189b9-787a-42df-9a07-b2f20dcffb95
  modified: 2026-09-30T02:52:39.499Z
---

For MDLM training data in `dllm-training/`, the foundation is Müller et al. 2022, "Transformers Can Do Bayesian Inference" (PFNs; code automl/TransformersCanDoBayesianInference). Its GP-mixture prior is vendored in `small_mdlm/vendor/pfn_priors.py`: x ~ U[0,1]^d, sigmoid → Bernoulli labels for classification, and x z-scored with the context rows.

**Why:** The user wants a principled, citable foundation rather than hand-designed task families. Decided 2026-09-30.

**How to apply:**
- New capabilities (multi-class for spirals, longer contexts, shorter length-scales, queries beyond the context) are documented extensions of the PFN prior, in the PFN/TabPFN lineage, e.g. multi-class by thresholding a continuous sample.
- Evaluate against classical baselines, the true p(y|x), shuffled-label controls and Dream. Do not benchmark against the earlier hand-built checkpoints (e.g. MDLM-BayesClassify2D-*).
- Regression should also move onto the PFN GP-mixture prior (`fast_gp_mix` without binarization).
- Related: [[dllm-training-tmp-location]].
