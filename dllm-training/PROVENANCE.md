# MDLM provenance and adaptations

The backbone in `small_mdlm/vendor/dit.py` is adapted from the official
[kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) implementation,
pinned to revision `c112c526d193436838c98d81455ee51f90309470`:
[original models/dit.py](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/models/dit.py).
The upstream Apache-2.0 license is retained in `small_mdlm/vendor/LICENSE`.

Retained architecture: token embeddings, bidirectional attention with rotary
positions, adaptive-layer-normalization DiT blocks, feed-forward layers, and
zero-initialized residual gates/output projection. This is a randomly
initialized small backbone, not a downloaded pretrained language model.

Local changes to the backbone:

- Replace FlashAttention with PyTorch scaled-dot-product attention and explicit
  split-half rotary positions; support key-padding masks for variable contexts.
- Use plain configuration dictionaries instead of OmegaConf; remove the
  Hugging Face Hub mixin, global JIT settings, and forced CUDA autocasting.
- Keep execution possible on CPU without installing FlashAttention.

The surrounding training loop, numeric tokenizer, synthetic episodes, and VUD
adapter are new local code. We do **not** vendor or reproduce the complete
upstream Lightning training/evaluation stack. This is a conditional numeric
specialization of MDLM, not a reproduction of its language-model benchmarks.

## Objective and sampling

`small_mdlm/model.py` implements an absorbing-mask denoising objective and
ancestral sampler following MDLM's SUBS parameterization. Only unknown answer
sign/digit tokens are eligible for masking. Context tokens, input covariates,
punctuation, and already revealed answer tokens remain fixed.

Training samples mask probability `t` uniformly in `[0.001, 1)`, masks each
eligible token independently, and weights masked-token cross-entropy by `1/t`.
Loss is normalized by the number of eligible tokens per episode, not the
realized mask count. Episodes with no masked tokens are retained. The small
lower cutoff truncates the continuous-time objective. We use no explicit time
conditioning (`sigma=0`) and constrain sign/digit vocabulary in both training
and sampling.

Sampling begins with all unknown answer tokens masked. At each reverse step,
each remaining mask is revealed with probability `(t-s)/t`; revealed tokens
are sampled from the model and then retained. The final step reveals every
remaining mask. Two unknown answers share one bidirectional sequence, allowing
joint sampling for parallel PP. The sampler does not guarantee that separately
queried joint and conditional distributions are statistically consistent.

## Relationship to the ICL-by-GD data

`data.py` preserves the independently implemented basic linear distribution
from the ICL-by-GD paper; it has not been silently changed. The training profile
in `small_mdlm/episodes.py` deliberately extends it to random bias, observation
noise, variable context sizes, and one or two queries sharing a latent function.
This supports VUD's conditional and joint sampling interfaces but is not an
exact reproduction of that paper's experimental setup.
