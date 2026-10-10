#!/bin/bash
# Setting 2 (seeded): labels from TabPFN v1's released model, each dataset seeded with 2-10 notebook-prior labels holding
# both classes (unseeded sampling collapsed to single-class datasets). 2.85M datasets in 4 shards on 2 GPUs -> bank ->
# MDLM with B6's recipe (GPU 0) -> VUD vanilla on logistic_regression and moons_1.
cd /home/js2723/PROJECTS/VUD/dllm-training
unset SLURM_PROCID LOCAL_RANK
export PYTHONUNBUFFERED=1
L=tmp/logs/tabpfn_model; SH=data_banks/tabpfn_model_seeded_shards; mkdir -p $L $SH
N=$(../.venv/bin/python -c "import numpy as np; print(np.load('data_banks/tabpfn_original_cls_3m/x.npy', mmap_mode='r').shape[0])")
Q=$(( (N + 3) / 4 ))
for s in 0 1 2 3; do
  lo=$((s * Q)); hi=$(( (s + 1) * Q < N ? (s + 1) * Q : N ))
  [ -f $SH/shard_$s.npz ] || CUDA_VISIBLE_DEVICES=$((s % 2)) ../.venv/bin/python tmp/scripts/tabpfn_model_generate.py $lo $hi $SH/shard_$s.npz \
    > $L/seeded_generate_$s.log 2>&1 &
done
wait
[ "$(ls $SH/shard_*.npz | wc -l)" = 4 ] || { echo 'generation failed'; exit 1; }
../.venv/bin/python - <<'PY' > $L/seeded_bank_join.log 2>&1 || exit 1
import json, numpy as np
from pathlib import Path
d = [np.load(f'data_banks/tabpfn_model_seeded_shards/shard_{s}.npz') for s in range(4)]
x, y = np.concatenate([a['x'] for a in d]).astype(np.float32), np.concatenate([a['y'] for a in d]).astype(np.int8)
order = np.random.default_rng(0).permutation(len(x))
out = Path('data_banks/tabpfn_model_seeded_cls_3m'); out.mkdir(parents=True, exist_ok=True)
np.save(out/'x.npy', x[order]); np.save(out/'y.npy', y[order]); np.save(out/'latent.npy', np.zeros(y.shape, np.float32))
(out/'meta.json').write_text(json.dumps({
    'prior': 'labels sampled sequentially from TabPFN v1 released model (temperature 1, 1 ensemble member, no power '
             'transform) on the inputs of data_banks/tabpfn_original_cls_3m, seeded with 2-10 notebook labels holding both classes',
    'generator': 'tmp/scripts/tabpfn_model_generate.py', 'datasets': int(len(x)), 'seq_len': int(x.shape[1]),
    'latent': 'placeholder zeros'}, indent=2))
print(f'bank: {len(x)} datasets, class-1 share {y.mean():.3f}, single-class datasets {(y.min(1) == y.max(1)).mean():.3f}')
PY
cat $L/seeded_bank_join.log
rm -f $SH/shard_*.npz && rmdir $SH
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python train.py --config configs/tabpfn_model_seeded_classify2d.json \
  --output outputs/tabpfn_model_seeded_classify2d --device cuda --workers 4 --worker-start-method spawn > $L/seeded_train.log 2>&1 || exit 1
cd /home/js2723/PROJECTS/VUD
export CUDA_VISIBLE_DEVICES=0 VUD_MDLM_DEVICE=cuda
CKPT=/home/js2723/PROJECTS/VUD/dllm-training/outputs/tabpfn_model_seeded_classify2d/checkpoint.pt
for dataset in moons_1 logistic_regression; do
  extra=(); [ "$dataset" = logistic_regression ] && extra=(--x_range "{'x1': [-15, 15, 0.2]}")
  .venv/bin/python run.py toy-classification --model_name "mdlm:$CKPT" --dataset_name "$dataset" --va_method vanilla \
    --model_temperature 1 --icl_sample_seed 0 --save_directory "MDLM-TabPFNModel-0.8M/va_vanilla" --resume 1 "${extra[@]}" \
    > tmp/logs/setting2_vud_$dataset.log 2>&1
done
.venv/bin/python dllm-training/tmp/scripts/settings_vud_summary.py > tmp/logs/settings_vud_summary.log 2>&1
echo 'SETTING 2 COMPLETE'
