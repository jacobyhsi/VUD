# Small masked-DLM training for VUD

A small MDLM-based training pipeline, numeric tokenizer, conditional/joint
sampler, and VUD regression adapter are now implemented. Start with
[TRAINING.md](TRAINING.md) for configurations and commands, and
[PROVENANCE.md](PROVENANCE.md) for the upstream source and adaptations.

The opt-in classification experiment in [CONTROLLED_GP_ABLATION.md](CONTROLLED_GP_ABLATION.md)
replaces a prior-bag GP member with explicit smoothness and clean/noisy
observation settings. Existing priors, the MDLM architecture, and VUD are unchanged.

The rest of this page documents the original standalone `data.py` generator,
which is preserved separately from the extended VUD training distribution.

## Provenance and defaults

The task distribution follows `create_reg_data` in the authors' code for
[Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677):
[original data.py](https://github.com/google-research/self-organising-systems/blob/master/transformers_learn_icl_by_gd/src/data.py).
Defaults follow their `constructed_token_setup.ipynb` configuration:

- 10 input coordinates, 10 demonstrations, one held-out query.
- A new weight vector per task: independent standard Gaussian coordinates.
- Inputs independently uniform on `[-input_range/2, input_range/2)`.
  Default `input_range=1` means `[-0.5, 0.5)`, not `[-1, 1)`.
- Outputs are `y = x @ w`, with no bias, noise, or distractor labels.

This is an independent NumPy implementation of the basic distribution, not a
copy of the JAX code or a bitwise reproduction of its random draws. Nonlinear,
OOD, distractor, and alternate-token experiments are deliberately deferred.
Arrays preserve their dimensions even for one-dimensional tasks. Each task
index identifies a deterministic sample within a seed and split; advance the
index to get fresh training tasks. A future multi-worker loader must assign
distinct indices to workers/ranks and save its position for resumption.

## Usage

Run scripts on the allocated compute node through `screen` and `srun`, following
the repository's `AGENTS.md`. From the repository root, inside that allocation:

```bash
.venv/bin/python dllm-training/data.py --seed 0 --task-index 0
.venv/bin/python dllm-training/data.py --input-size 1 --context-size 3
.venv/bin/python -m unittest discover -s dllm-training -p 'test_*.py' -v
```

The standalone `data.py` generator needs only NumPy. Training and the full test
suite additionally use PyTorch, einops, and the existing VUD dependencies.

For Python callers in this directory:

```python
from data import LinearRegressionTasks

tasks = LinearRegressionTasks(seed=0, split='train')
task = tasks.sample(0)
inputs = task.model_inputs()  # context_x, context_y, query_x only
target = task.query_y         # supervision; do not put into the clean prompt
```

`task.weights` is diagnostic ground truth, not a model input.
`task.constructed_tokens()` provides the paper-style numeric array
`[(x_1,y_1), ..., (x_n,y_n), (x_query,0)]`. Its zero placeholder is **not** a
discrete mask token; the training pipeline uses its own explicit mask/padding
tokens and numeric serialization. The preview CLI prints labels
and weights in separate fields for inspection, not as a ready-to-train prompt.
