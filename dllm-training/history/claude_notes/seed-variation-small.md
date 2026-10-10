---
name: seed-variation-small
description: "User finds VUD results barely change across icl_sample_seed values; don't propose extra-seed runs"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ca6189b9-787a-42df-9a07-b2f20dcffb95
  modified: 2026-10-01T02:43:11.430Z
---

Don't propose rerunning VUD experiments with more `icl_sample_seed` values (e.g. seeds 1 and 2) to firm up a comparison.

**Why:** In the user's experience, results barely differ between seeds, so extra seeds aren't worth the GPU time. They said so on 2026-10-01 when I suggested more seeds for logistic_regression.

**How to apply:** Treat single-seed (seed 0) VUD results as representative. Spend verification effort on other checks instead, such as Bayes-reference (mode-1) comparisons or estimator diagnostics. Related: [[pfn-prior-foundation]].
