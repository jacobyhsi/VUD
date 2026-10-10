# Small numeric MDLM: training and VUD

This prototype trains a small, randomly initialized MDLM backbone on synthetic
linear-regression episodes. It can plug into VUD's **toy-regression vanilla and
parallel_pp** paths. It is not a general-purpose Dream replacement: QA,
classification, and the other parallel estimators are not supported.

Read [PROVENANCE.md](PROVENANCE.md) for the upstream source and modifications.
Runtime dependencies are NumPy, PyTorch with scaled-dot-product attention, and
einops; VUD inference additionally uses the repository's existing dependencies.
No FlashAttention or pretrained tokenizer is required.

## Starting configuration

`model.cond_dim` (the DiT's timestep-conditioning width) defaults to 0 when a
config omits it. We always pass sigma=0, as MDLM does with
`time_conditioning: False`, so that input is constant; with 0 each block's
adaLN keeps only its learned, zero-initialized scale/shift/gate bias. In a
size ablation on the broad linear prior, `cond_dim` 32 matched 128. Older
configs set it explicitly, and their checkpoints load unchanged.

`configs/linear_vud.json` is a proposed starting point, not a tuned result:

| Setting | Value |
| --- | --- |
| Backbone | 4 blocks, width 128, 4 heads, dropout 0 |
| Task | One-dimensional `y = x*w + b + noise` |
| Function per episode | `w ~ N(0, 2^2)`, `b ~ N(0, 3^2)` |
| Covariates | Independent uniform `[-15, 15)` |
| Observation noise | Independent `N(0, 2^2)` per row |
| Observed examples | Uniformly sampled count from 3 through 30 |
| Hidden answers | One or two, sharing the same latent function |
| Number encoding | Explicit sign, three integer digits, two decimals |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0, gradient clip 1 |
| Training | 10,000 steps, batch size 32, seed 0, fresh tasks each step |
| Validation | Every 250 steps, 32 fixed held-out episodes |
| Sampling | 32 reverse steps, temperature 1 by default |

This training distribution differs from VUD's fixed linear function and
Gaussian-distributed ICL covariates. That evaluation is a distribution shift,
not a claim of matched training/test data. Vanilla with D=30 adds an observed
auxiliary row, giving 31 observations, also just beyond the trained context
count; use D=15 initially or deliberately extend the training profile.

## Train

Run scripts inside your Slurm allocation using `screen` and `srun`, as required
by the repository's `AGENTS.md`. From the repository root, once on the compute
node and with a suitable GPU selected:

```bash
.venv/bin/python dllm-training/train.py \
  --config dllm-training/configs/linear_vud.json \
  --output dllm-training/outputs/linear_seed0 \
  --device cuda
```

Repeat with `--resume` to continue the same configuration. The output contains
`config.json`, `metrics.jsonl`, and `checkpoint.pt` (latest checkpoint, replaced
atomically at validation intervals). Fresh runs reject nonempty output folders.
Checkpointing saves optimizer, data-step position, and random states. Exact
resumption has been tested on CPU, not across devices or different hardware.
There is no EMA, learning-rate scheduler, distributed trainer, or early stopping.

## Use a trained checkpoint in VUD

These examples require a checkpoint produced by training above. A successful
smoke checkpoint is not sufficient evidence of useful predictions.

```bash
export VUD_MDLM_DEVICE=cuda
CHECKPOINT=/home/js2723/PROJECTS/VUD/dllm-training/outputs/linear_seed0/checkpoint.pt

.venv/bin/python run.py toy-regression \
  --dataset_name linear_regression \
  --model_name "mdlm:$CHECKPOINT" \
  --D_size 15 --va_method vanilla \
  --save_directory small-mdlm-linear-seed0/va_vanilla

.venv/bin/python run.py toy-regression \
  --dataset_name linear_regression \
  --model_name "mdlm:$CHECKPOINT" \
  --D_size 15 --va_method parallel_pp --num_joint_samples 100 \
  --save_directory small-mdlm-linear-seed0/va_parallel_pp_n100
```

The adapter defaults to CPU unless `VUD_MDLM_DEVICE` is set. It uses the numeric
rows from VUD prompts rather than natural-language instructions. Existing
regression defaults (including IQR standard deviation and one-pair trimming)
remain unchanged. Both single-answer and joint-answer sampling use the same
checkpoint; weights stay fixed during VUD inference. Checkpoint input dimension
must match the dataset. Load only trusted local checkpoints.

## Verification and scientific limitations

Run lightweight tests on an allocated compute node:

```bash
.venv/bin/python -m unittest discover -s dllm-training -p 'test_*.py' -v
.venv/bin/python dllm-training/train.py \
  --config dllm-training/configs/smoke.json \
  --output tmp/mdlm-smoke-example --device cpu
```

Unit tests cover deterministic episodes, hidden-target isolation, numeric
boundaries, loss/backpropagation, clamped context, deterministic sampling,
padding attention, and the VUD response format. CPU integration checks also
exercise checkpoint resume and the actual vanilla/parallel_pp runners.

Numbers are rounded to two decimal places, including covariates. Encodings
outside the representable range (approximately -999.99 to 999.99) fail rather
than clip. Joint sampling is sequential across requested samples in the adapter;
batching this could improve throughput. Repeated quantized predictions can
produce zero IQR; existing VUD estimator behavior has not been changed.

Before interpreting uncertainty plots, validate regression accuracy against
simple fitted-regression baselines, dependence on demonstrations (including a
label-shuffle control), predictive calibration, joint/conditional consistency,
and behavior under VUD's covariate distribution. The trainer reports masked
validation loss, single-sample squared error against noisy targets, and target
quantization error; these alone do not establish any of those properties.
