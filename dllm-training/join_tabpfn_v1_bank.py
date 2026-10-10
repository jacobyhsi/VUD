"""Join TabPFN-v1 shards (generate_tabpfn_v1_prior_data.py / generate_tabpfn_v1_model_labels.py) into a
classification_pfn bank: x (N, 64, 2) float32, y (N, 64) int8, latent (N, 64) float32, meta.json.

TabPFN gives no true p(y|x), so latent is a 0 placeholder (used only by training-time validation metrics against the
true p; the training loss uses the labels). Datasets are shuffled across shards before the bank's holdout split.
Usage: python join_tabpfn_v1_bank.py <shard dir with shard_*.npz> <bank dir> "<description of the data>"
"""
import json
import sys
from pathlib import Path

import numpy as np

shard_dir, out, description = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
shards = sorted(shard_dir.glob('shard_*.npz'))
if not shards:
    raise SystemExit(f'no shard_*.npz in {shard_dir}')
data = [np.load(f) for f in shards]
x = np.concatenate([d['x'] for d in data]).astype(np.float32)
y = np.concatenate([d['y'] for d in data]).astype(np.int8)
order = np.random.default_rng(0).permutation(len(x))
out.mkdir(parents=True, exist_ok=True)
np.save(out/'x.npy', x[order])
np.save(out/'y.npy', y[order])
np.save(out/'latent.npy', np.zeros(y.shape, np.float32))
(out/'meta.json').write_text(json.dumps({
    'prior': description, 'shards': [s.name for s in shards], 'datasets': int(len(x)), 'seq_len': int(x.shape[1]),
    'latent': 'placeholder zeros (no true p(y|x))'}, indent=2))
print(f'bank {out}: {len(x)} datasets, class-1 share {y.mean():.3f}, '
      f'single-class datasets {(y.min(1) == y.max(1)).mean():.3f}, 1-feature share {(np.abs(x[:, :, 1]).sum(1) == 0).mean():.2f}')
