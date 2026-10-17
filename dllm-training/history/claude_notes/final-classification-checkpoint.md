---
name: final-classification-checkpoint
description: "The user's adopted final MDLM classification checkpoint (2026-10-07) and the recipe that produced it"
metadata:
  node_type: memory
  type: project
  originSessionId: ca6189b9-787a-42df-9a07-b2f20dcffb95
  modified: 2026-10-07T10:16:02.223Z
---

On 2026-10-07 the user adopted `dllm-training/outputs/tabpfn_controlled_gp50_geometry_1d_layout75/checkpoint_step_020000.pt` as the final classification checkpoint. Its VUD results name is `MDLM-TabPFN-ControlledGP50-Geometry1D-Layout75-0.8M`. Since the 2026-10-11 move, `outputs/` and `data_banks/` are gone. The only copy is `dllm-training/history/checkpoints/final_classification_controlled_gp50_geometry_1d_layout75.pt`.

Recipe: 0.8M DiT MDLM, 20k steps. Prior: 50% controlled GP (`configs/controlled_gp_prior.json`) + 50% TabPFN v1 MLP (non-causal). Paired geometry episodes (`small_mdlm/tabpfn_episodes.py`) with one_d_prob 0.5 and layout_prob 0.75, bank `data_banks/controlled_gp50_episode_pairs_1d_layout75_3m/geometry`.

What mattered, in order: the prior must have clean curved boundaries plus far-field disagreement; training episodes must match VUD's query (z next to x, far queries); structured input layouts fixed the far-field overconfidence on moons. Model size and training length did not help.

**Why:** many rounds were needed to get here. Revisit this before starting new classification prior or training work.
**How to apply:** use it as the baseline for further classification experiments. Score new runs with `tmp/scripts/conditional_reference.py` (now archived in `dllm-training/history/scripts/dllm_training_tmp/`; copy it back into `dllm-training/tmp/scripts/` and fix the old `/home/js2723/...` paths before use) (SCORE_ONLY), `farfield_breakdown.py` and `tabpfn_lineage_summary.py`. Related: [[prior-evidence-check]], [[chain-follow-up-steps]].
