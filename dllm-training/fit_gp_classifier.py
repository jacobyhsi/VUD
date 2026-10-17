"""Fit a standard GP classifier's hyperparameters to a classification bank (the GP counterpart of an MDLM trained
on the same bank, e.g. setting 2's TabPFN-generated data).

Model: scikit-learn's GaussianProcessClassifier with kernel ConstantKernel(amplitude) * RBF(lengthscale), i.e. latent
f ~ GP(0, amplitude * exp(-|x - x'|^2 / (2 lengthscale^2))), p(y = 1 | f) = sigmoid(f), Laplace inference. Each
dataset's inputs are z-scored with its own points (normalize_by_context, as the MDLM sees them; the zero x2 of 1-D
datasets stays 0). Training maximises the Laplace log marginal likelihood summed over the bank's training split
(all but the last bank_holdout datasets, as for the MDLM) by Adam on (log amplitude, log lengthscale) over shuffled
minibatches, one pass over the split. The Newton iterations are unrolled under autograd, so the gradient includes the
mode's dependence on the hyperparameters. The result is checked on the validation split, including a small grid
around the optimum.
Usage: python fit_gp_classifier.py <bank dir> <out.json> [--batch=1024] [--device=cuda] [--check]
(--check compares the batched log marginal likelihood with scikit-learn's on a few datasets and exits)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BANK_HOLDOUT = 20000
NEWTON_STEPS = 20


def zscore(x):
    """normalize_by_context with every point as context, vectorised over datasets: (B, n, d) -> (B, n, d)."""
    mean, std = x.mean(1, keepdims=True), x.std(1, keepdims=True)
    z = (x - mean) / np.where(std > 1e-8, std, 1.0)
    return np.clip(np.where(std > 1e-8, z, 0.0), -50.0, 50.0)


def laplace_log_marginal(x, y, log_amplitude, log_lengthscale, steps=NEWTON_STEPS):
    """Batched Laplace log marginal likelihood, GPML Algorithm 3.1 exactly as in scikit-learn's binary GPC.

    x (B, n, d), y (B, n) in {0, 1}; returns (B,)."""
    sq = torch.cdist(x, x) ** 2
    K = log_amplitude.exp() * torch.exp(-0.5 * sq / (2 * log_lengthscale).exp())
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    f = torch.zeros_like(y)
    for _ in range(steps):
        pi = torch.sigmoid(f)
        W = pi * (1 - pi)
        W_sr = W.sqrt()
        W_sr_K = W_sr[..., :, None] * K
        L = torch.linalg.cholesky(eye + W_sr_K * W_sr[..., None, :])
        b = W * f + (y - pi)
        a = b - W_sr * torch.cholesky_solve((W_sr_K @ b[..., None]), L)[..., 0]
        f = (K @ a[..., None])[..., 0]
    return (-0.5 * (a * f).sum(-1) - torch.nn.functional.softplus(-(2 * y - 1) * f).sum(-1)
            - torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1))


def load(bank, rows, device):
    x = torch.tensor(zscore(np.asarray(bank['x'][rows], dtype=np.float64)), device=device)
    y = torch.tensor(np.asarray(bank['y'][rows], dtype=np.float64), device=device)
    return x, y


def check(bank):
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    amplitude, lengthscale = 2.0, 0.7
    x, y = load(bank, np.arange(5), 'cpu')
    ours = laplace_log_marginal(x, y, torch.tensor(np.log(amplitude), dtype=torch.float64),
                                torch.tensor(np.log(lengthscale), dtype=torch.float64), steps=100)
    theirs = [GaussianProcessClassifier(ConstantKernel(amplitude, 'fixed') * RBF(lengthscale, 'fixed'))
              .fit(x[i].numpy(), y[i].numpy()).log_marginal_likelihood_value_ for i in range(5)]
    for i in range(5):
        print(f'dataset {i}: batched {ours[i].item():.6f}  scikit-learn {theirs[i]:.6f}')
    return np.allclose(ours.numpy(), theirs, atol=1e-6)


@torch.no_grad()
def mean_log_marginal(bank, rows, log_amplitude, log_lengthscale, batch, device):
    total = 0.0
    for i in range(0, len(rows), batch):
        x, y = load(bank, rows[i:i + batch], device)
        total += laplace_log_marginal(x, y, log_amplitude, log_lengthscale).sum().item()
    return total / len(rows)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    bank_dir, out = Path(args[0]), Path(args[1]) if len(args) > 1 else None
    batch, device = int(opts.get('batch', 1024)), opts.get('device', 'cuda')
    bank = {k: np.load(bank_dir/f'{k}.npy', mmap_mode='r') for k in ('x', 'y')}
    if '--check' in sys.argv:
        sys.exit(0 if check(bank) else 1)

    total = len(bank['y'])
    train = np.random.default_rng(0).permutation(total - BANK_HOLDOUT)
    validation = np.arange(total - BANK_HOLDOUT, total - BANK_HOLDOUT // 2)
    params = torch.zeros(2, dtype=torch.float64, device=device, requires_grad=True)   # log amplitude, log lengthscale
    optimizer = torch.optim.Adam([params], lr=0.02)
    steps = (len(train) + batch - 1) // batch
    average, averaged, start = torch.zeros(2, dtype=torch.float64), 0, time.time()
    for step in range(steps):
        x, y = load(bank, np.sort(train[step * batch:(step + 1) * batch]), device)
        loss = -laplace_log_marginal(x, y, params[0], params[1]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= int(0.9 * steps):                  # average the last 10% of steps to remove minibatch noise
            average += params.detach().cpu()
            averaged += 1
        if step % 100 == 0 or step == steps - 1:
            a, l = params.detach().exp().tolist()
            print(f'step {step}/{steps}  mean log marginal {-loss.item():.4f}  amplitude {a:.4f}  '
                  f'lengthscale {l:.4f}  ({time.time() - start:.0f}s)', flush=True)
    log_amplitude, log_lengthscale = (average / averaged).to(device)

    grid = {}
    for da in (-1, 0, 1):
        for dl in (-1, 0, 1):
            grid[f'amplitude x{1.25 ** da:.2f}, lengthscale x{1.25 ** dl:.2f}'] = mean_log_marginal(
                bank, validation, log_amplitude + da * np.log(1.25), log_lengthscale + dl * np.log(1.25), batch, device)
    best = max(grid, key=grid.get)
    result = {
        'kernel': 'ConstantKernel(amplitude) * RBF(lengthscale) on per-context z-scored inputs',
        'amplitude': log_amplitude.exp().item(), 'lengthscale': log_lengthscale.exp().item(),
        'bank': str(bank_dir), 'train_datasets': int(len(train)), 'batch': batch, 'steps': steps,
        'validation_mean_log_marginal': grid['amplitude x1.00, lengthscale x1.00'],
        'validation_grid_x1.25': grid, 'fitted_is_grid_best': best == 'amplitude x1.00, lengthscale x1.00',
    }
    print(json.dumps(result, indent=2))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
