"""Setting 2 data: labels sampled from TabPFN v1's released MODEL (not its prior), on the notebook datasets' inputs.

For each dataset (x, y from the notebook-prior bank, 64 points) the first m ~ U{2..10} points keep their
notebook-prior labels as a SEED containing both classes (if the first m are single-class, the first later point of the
other class is swapped into the seed; points within a dataset are exchangeable). The remaining labels are drawn
sequentially from TabPFN's predictive, y_k ~ p_TabPFN(y | x_k, (x_i, y_i)_{i<k}) for k = m+1..64, each sampled label
joining the context. Unseeded sampling collapses: TabPFN was only trained on contexts with both classes, gives an absent
class probability ~0, and 72% of unseeded datasets ended single-class.
TabPFN is run batched through its transformer with its own inference preprocessing for one ensemble member without
power transform / feature or class shifts (TabPFNClassifier(N_ensemble_configurations=1, no_preprocess_mode=True,
feature_shift_decoder=False, multiclass_decoder=None)): normalize_data and remove_outliers over the context,
constant context features removed, rescaling by used features, zero-padding to 100. Temperature 1 (TabPFN's raw
predictive; its default additionally divides logits by 0.8). For contexts of < 2 points the normalisation also uses
the query (TabPFN's statistics are undefined with one point).
Needs a GPU and TabPFN v1 (setup_tabpfn_v1.sh); see TABPFN_V1.md.
Usage: python generate_tabpfn_v1_model_labels.py <first> <last, exclusive> <out.npz> [--source=<bank dir>] [--check]
(--source defaults to data_banks/tabpfn_original_cls_3m; join shards with join_tabpfn_v1_bank.py;
--check compares the batched path with TabPFNClassifier itself and exits)
"""
import os
import sys
from pathlib import Path

for _var in ('SLURM_PROCID', 'LOCAL_RANK'):                                 # TabPFN must not start distributed mode
    os.environ.pop(_var, None)

import numpy as np                                                           # noqa: E402
import torch                                                                 # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from small_mdlm.tabpfn_v1 import setup                                       # noqa: E402
REPO = setup()
from tabpfn import TabPFNClassifier                                          # noqa: E402
from tabpfn.utils import normalize_data, remove_outliers                     # noqa: E402

dev = 'cuda'
FIRST, LAST, OUT = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
CHECK = '--check' in sys.argv
SOURCE = next((Path(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--source=')),
              ROOT/'data_banks'/'tabpfn_original_cls_3m')
BATCH, MAX_FEATURES = 2048, 100
clf = TabPFNClassifier(device=dev, base_path=str(REPO/'tabpfn'), N_ensemble_configurations=1, no_preprocess_mode=True,
                       feature_shift_decoder=False, multiclass_decoder=None)
model = clf.model[2].to(dev).eval()


@torch.no_grad()
def prefix_probs(x, y, k):
    """p(y_k = 1 | first k points as context) for a batch: x (B, >=k+1, 2), y (B, >=k) -> (B,)."""
    xs = x[:, :k + 1].transpose(0, 1).float()                               # (T, B, 2), T = k + 1, query last
    pos = k if k >= 2 else -1
    xs = normalize_data(xs, normalize_positions=pos)
    xs = remove_outliers(xs, normalize_positions=pos)
    ctx = x[:, :k].transpose(0, 1)
    used = (ctx.amax(0) - ctx.amin(0)) > 0 if k >= 2 else torch.ones(x.shape[0], 2, dtype=torch.bool, device=dev)
    xs = xs * used[None]                                                    # remove constant context features
    swap = ~used[:, 0] & used[:, 1]                                         # pack kept features to the front
    xs[:, swap] = xs[:, swap].flip(-1)
    n_used = used.sum(1).clamp(min=1).float()
    xs = xs / (n_used / MAX_FEATURES)[None, :, None]
    xs = torch.cat([xs, torch.zeros(*xs.shape[:2], MAX_FEATURES - 2, device=dev)], -1)
    ys = torch.cat([y[:, :k], torch.zeros(x.shape[0], 1, device=dev)], 1).transpose(0, 1).float()
    with torch.autocast('cuda', dtype=torch.float16):
        out = model((None, xs, ys), single_eval_pos=k)[-1, :, :2].float()
    return torch.softmax(out, -1)[:, 1]


bank_x = np.load(SOURCE/'x.npy', mmap_mode='r')
bank_y = np.load(SOURCE/'y.npy', mmap_mode='r')
SEED_MIN, SEED_MAX = 2, 10
if CHECK:                                                                   # batched path vs TabPFNClassifier itself
    x = torch.tensor(np.array(bank_x[FIRST:FIRST + 8]), device=dev)
    y = (torch.rand(8, 64, device=dev) > 0.5).float()
    for k in (5, 20, 40):
        mine = prefix_probs(x, y, k).cpu().numpy()
        ref = []
        for b in range(8):
            xc, yc = x[b, :k].cpu().numpy(), y[b, :k].cpu().numpy().astype(int)
            xc_used = xc[:, np.ptp(xc, 0) > 0]
            if len(np.unique(yc)) < 2 or xc_used.shape[1] == 0:
                ref.append(np.nan); continue
            fitted = clf.fit(xc, yc, overwrite_warning=True)
            lg = fitted.predict_proba(x[b, k:k + 1].cpu().numpy(), return_logits=True)[0] * 0.8
            ref.append(np.exp(lg[1]) / np.exp(lg).sum())
        print(f'k={k}: batched {mine.round(3)}\n      classifier {np.array(ref).round(3)}')
    sys.exit()

xs_out, ys_out, seeds_out = [], [], []
rng = np.random.default_rng(FIRST)
torch.manual_seed(FIRST)
for s in range(FIRST, LAST, BATCH):
    xb, yb = np.array(bank_x[s:min(s + BATCH, LAST)]), np.array(bank_y[s:min(s + BATCH, LAST)]).astype(np.float32)
    m = rng.integers(SEED_MIN, SEED_MAX + 1, len(xb))
    for b in range(len(xb)):                                                # seed must contain both classes
        if yb[b, :m[b]].min() == yb[b, :m[b]].max():
            j = m[b] + int(np.argmax(yb[b, m[b]:] != yb[b, 0]))
            xb[b, [m[b] - 1, j]], yb[b, [m[b] - 1, j]] = xb[b, [j, m[b] - 1]], yb[b, [j, m[b] - 1]]
    x, y = torch.tensor(xb, device=dev), torch.tensor(yb, device=dev)
    seeded = torch.arange(x.shape[1], device=dev)[None] < torch.tensor(m, device=dev)[:, None]
    for k in range(SEED_MIN, x.shape[1]):
        sample = (torch.rand(len(x), device=dev) < prefix_probs(x, y, k)).float()
        y[:, k] = torch.where(seeded[:, k], y[:, k], sample)
    seeds_out.append(m)
    xs_out.append(x.cpu().numpy()); ys_out.append(y.to(torch.int8).cpu().numpy())
    if (s - FIRST) // BATCH % 50 == 0:
        print(f'{s + len(x) - FIRST}/{LAST - FIRST} datasets', flush=True)
np.savez(OUT, x=np.concatenate(xs_out), y=np.concatenate(ys_out), seed_size=np.concatenate(seeds_out))
print('saved', OUT)
