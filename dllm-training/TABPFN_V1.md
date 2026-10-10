# TabPFN v1 experiments (three settings)

Three reference points for VUD on the classification toys, all built on TabPFN v1
(automl/TabPFN, tag v1.0.0):

1. **MDLM (notebook prior)**: an MDLM trained on data from TabPFN v1's original prior code.
2. **MDLM (TabPFN-generated)**: an MDLM trained on labels sampled from TabPFN v1's released model.
3. **TabPFN v1**: the released classifier itself, used as the VUD model.

## Setup

```bash
cd dllm-training
./setup_tabpfn_v1.sh   # clones TabPFN v1.0.0 (code, prior, checkpoint) into third_party/ and installs ConfigSpace
```

`third_party/` is git-ignored. Set `TABPFN_V1_DIR` to use an existing clone. A small compatibility
shim (`small_mdlm/tabpfn_v1.py`) lets TabPFN v1's 2022 code import on current PyTorch; TabPFN's own
code is not modified.

## 1. MDLM (notebook prior)

The prior is TabPFN's own notebook recipe (`tabpfn/PriorFittingCustomPrior.ipynb`): its prior bag of
GP (about 4% of datasets) and MLP/SCM (causal and non-causal), with the differentiable hyperpriors.
The only changes are what the MDLM needs: binary labels, 1–2 features, 64 points per dataset, and a
context/test split of 10–30 points for TabPFN's class-compatibility check. Generation runs on CPU,
at about 400 datasets per second per process.

```bash
for s in $(seq 1 24); do   # seed 0 is reserved for the test set
  python generate_tabpfn_v1_prior_data.py 125000 data_banks/tabpfn_original_shards/shard_$s.npz $s &
done; wait
python join_tabpfn_v1_bank.py data_banks/tabpfn_original_shards data_banks/tabpfn_original_cls_3m \
  "TabPFN v1 original prior (notebook recipe); binary, 1-2 features, 64 points"
python train.py --config configs/tabpfn_original_classify2d.json --output outputs/tabpfn_original_classify2d \
  --device cuda --workers 4 --worker-start-method spawn
```

## 2. MDLM (TabPFN-generated)

Labels come from TabPFN v1's released model, on the inputs of the notebook-prior bank above:
- **Seed:** the first 2–10 points of each dataset keep their notebook labels, always including both classes.
- **Sampling:** the remaining labels are drawn one at a time from TabPFN's predictive, at temperature 1 with one
  ensemble member and no power transform. Each sampled label joins the context before the next is drawn.
- **Why seeded:** unseeded sampling collapses. TabPFN gives a class absent from the context a probability of about
  0, so 72% of datasets came out single-class.

`--check` verifies that the batched path matches `TabPFNClassifier`.

```bash
python generate_tabpfn_v1_model_labels.py 0 8 /dev/null --check
N=2852538; Q=$(( (N + 3) / 4 ))
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((s % 2)) python generate_tabpfn_v1_model_labels.py $((s * Q)) $(( (s + 1) * Q < N ? (s + 1) * Q : N )) \
    data_banks/tabpfn_model_seeded_shards/shard_$s.npz &
done; wait
python join_tabpfn_v1_bank.py data_banks/tabpfn_model_seeded_shards data_banks/tabpfn_model_seeded_cls_3m \
  "labels sampled from TabPFN v1 (temperature 1), seeded with 2-10 notebook labels holding both classes"
python train.py --config configs/tabpfn_model_seeded_classify2d.json --output outputs/tabpfn_model_seeded_classify2d \
  --device cuda --workers 4 --worker-start-method spawn
```

## VUD on the toys (from the repository root)

```bash
export VUD_MDLM_DEVICE=cuda VUD_TABPFN_DEVICE=cuda
for ds in moons_1 logistic_regression; do
  extra=(); [ "$ds" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
  common=(--dataset_name $ds --va_method vanilla --model_temperature 1 --icl_sample_seed 0 --resume 1 "${extra[@]}")
  python run.py toy-classification --model_name mdlm:$PWD/dllm-training/outputs/tabpfn_original_classify2d/checkpoint.pt \
    --save_directory MDLM-TabPFNNotebook-0.8M/va_vanilla "${common[@]}"
  python run.py toy-classification --model_name mdlm:$PWD/dllm-training/outputs/tabpfn_model_seeded_classify2d/checkpoint.pt \
    --save_directory MDLM-TabPFNModel-0.8M/va_vanilla "${common[@]}"
  python run.py toy-classification --model_name tabpfn:v1 --save_directory TabPFN-v1/va_vanilla "${common[@]}"
done
```

`tabpfn:v1` is TabPFN as published: 3 ensemble members, logits divided by 0.8. `tabpfn:v1-t1`
undoes the 0.8 temperature.

## Results so far (bits; near / far from the data)

| | LR AU error | LR EU | moons_1 AU error | moons_1 EU |
|---|---|---|---|---|
| MDLM (notebook prior) | 0.06 / 0.35 | 0.02 / 0.06 | 0.32 / 0.20 | 0.04 / 0.07 |
| MDLM (TabPFN-generated) | 0.20 / 0.24 | 0.05 / 0.15 | 0.23 / 0.26 | 0.06 / 0.19 |
| TabPFN v1 | 0.29 / 0.23 | 0.07 / 0.38 | 0.14 / 0.13 | 0.19 / 0.40 |

**What this shows:**
- **VUD works on a strong PFN:** with TabPFN v1, far-field uncertainty goes into EU.
- **The notebook-prior MDLM barely reacts to distance from the data.**
- **Training on TabPFN-generated data recovers TabPFN's TU shape,** but only about half of its far-field EU.
