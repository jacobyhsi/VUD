"""Evaluate a general 1-D regression checkpoint, including task types it never saw.

Suites (one hidden query per task):
  training_family   the checkpoint's own task prior                     (D = 15)
  vud_linear        VUD linear_regression: y = -x + 3 + N(0, 2^2), x ~ N(1, 2^2)  (D = 15)
  vud_gaps          VUD gaps generator (three modes; lines/noise per mode),
                    queries drawn from the same modes                   (D = 30)
  heldout_step      piecewise-constant f with 1-3 jumps                (never trained on)
  heldout_poly      cubic polynomial f                                 (never trained on)
  heldout_heavytail smooth f with Student-t (3 dof) noise              (never trained on)
Held-out suites use uniform or clustered x on [-15, 15) and D = 15.

For each suite: MSE of the MDLM sample mean against the true E[y|x]; Gaussian
negative log-likelihood of the noisy target (sample mean/variance); 90%
interval coverage; a shuffled-label control; and a GP regression baseline
(sklearn, RBF + linear + white-noise kernel) fitted to the same context.
Never trains.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel

from evaluate import predictions
from small_mdlm.backend import NumericMDLM
from small_mdlm.episodes import EpisodeConfig, episode

GAPS = json.load(open(Path(__file__).resolve().parents[1]/'datasets_toy_regression/gaps/info.json'))


def _context_x(rng, n):
    if rng.random() < 0.5:
        return rng.uniform(-15, 15, n)
    return rng.normal(rng.uniform(-7, 7), np.exp(rng.uniform(np.log(1), np.log(5))), n)


def make_task(suite, prior, seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7, index, len(suite)]))
    if suite == 'training_family':
        t = episode(replace(prior, context_min=15, context_max=15, max_queries=1), seed, index, 'test')
        return t['x'][:, 0], t['y'], t['mean'], t['std']
    if suite == 'vud_linear':
        x = np.append(rng.normal(1, 2, 15), rng.uniform(-15, 15))
        mean, std = -x + 3, np.full(16, 2.0)
    elif suite == 'vud_gaps':
        counts = np.array(GAPS['num_features_per_mode'], dtype=float)
        mode = rng.choice(len(counts), size=31, p=counts / counts.sum())
        x = rng.normal(np.array(GAPS['mode_means'])[mode], np.array(GAPS['mode_stds'])[mode])
        mean = x * np.array(GAPS['mode_coeffs'])[mode] + np.array(GAPS['mode_biases'])[mode]
        std = np.array(GAPS['noise_stds'])[mode]
    else:
        x = np.append(_context_x(rng, 15), rng.uniform(-15, 15))
        if suite == 'heldout_step':
            cuts = np.sort(rng.uniform(-12, 12, rng.integers(1, 4)))
            levels = rng.normal(0, 8, len(cuts) + 1)
            mean = levels[np.searchsorted(cuts, x)]
        elif suite == 'heldout_poly':
            c = rng.normal(size=4) * np.array([5, 1, 0.1, 0.01])
            mean = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3
        else:
            w, length = rng.normal(0, 1), np.exp(rng.uniform(np.log(2), np.log(10)))
            mean = w * x + 4 * np.sin(x / length + rng.uniform(0, 2 * np.pi))
        scale = np.exp(rng.uniform(np.log(0.1), np.log(2)))
        if suite == 'heldout_heavytail':
            noise, std = rng.standard_t(3, len(x)) * scale, np.full(len(x), scale * np.sqrt(3))
            return x, mean + noise, mean, std
        std = np.full(len(x), scale)
    return x, mean + std * rng.normal(size=len(x)), mean, std


def gp_baseline(x, y, xq):
    kernel = ConstantKernel(1.0) * RBF(3.0) + DotProduct(1.0) + WhiteKernel(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=0).fit(x[:, None], y)
    mean, std = gp.predict(np.array([[xq]]), return_std=True)
    return float(mean[0]), float(std[0])


def scores(mean, var, low, high, truth, noisy):
    var = np.maximum(var, 1e-6)
    return {'mse_vs_true_mean': float(np.mean((mean - truth) ** 2)),
            'gaussian_nll': float(np.mean(0.5 * np.log(2 * np.pi * var) + (noisy - mean) ** 2 / (2 * var))),
            'coverage90': float(np.mean((noisy >= low) & (noisy <= high)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tasks', type=int, default=200)
    parser.add_argument('--draws', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh evaluation directory')
    torch.set_num_threads(4)
    backend = NumericMDLM(args.checkpoint, args.device)
    prior, tok = EpisodeConfig(**backend.config['data']), backend.tokenizer
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'tasks_per_suite': args.tasks, 'draws': args.draws, 'seed': args.seed, 'suites': {}}
    suites = ['training_family', 'vud_linear', 'vud_gaps', 'heldout_step', 'heldout_poly', 'heldout_heavytail']
    for suite in suites:
        tasks = [make_task(suite, prior, args.seed, i) for i in range(args.tasks)]
        truth = np.array([t[2][-1] for t in tasks])
        noisy = np.array([t[1][-1] for t in tasks])
        rng = np.random.default_rng(args.seed)
        encoded, shuffled, gp = [], [], []
        for x, y, _, _ in tasks:
            x, y = np.round(x, tok.decimals), np.round(y, tok.decimals)
            y_hidden = np.append(y[:-1], np.nan)
            encoded.append(tok.encode(x[:, None], y_hidden, [len(y) - 1], hide=True))
            shuffled.append(tok.encode(x[:, None], np.append(rng.permutation(y[:-1]), np.nan),
                                       [len(y) - 1], hide=True))
            gp.append(gp_baseline(x[:-1], y[:-1], x[-1]))
        draws = predictions(backend, encoded, args.draws, args.batch_size, args.seed)
        shuffled_draws = predictions(backend, shuffled, args.draws, args.batch_size, args.seed)
        gp = np.array(gp)
        low, high = np.quantile(draws, [0.05, 0.95], axis=1)
        result = {
            'mdlm': scores(draws.mean(1), draws.var(1, ddof=1), low, high, truth, noisy),
            'mdlm_shuffled_labels': {'mse_vs_true_mean': float(np.mean((shuffled_draws.mean(1) - truth) ** 2))},
            'gp_regression': scores(gp[:, 0], gp[:, 1] ** 2, gp[:, 0] - 1.645 * gp[:, 1],
                                    gp[:, 0] + 1.645 * gp[:, 1], truth, noisy),
            'true_noise_var_mean': float(np.mean([t[3][-1] ** 2 for t in tasks])),
            'mdlm_predictive_var_mean': float(draws.var(1, ddof=1).mean())}
        report['suites'][suite] = result
        print(json.dumps({suite: result}), flush=True)
        (args.output/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Evaluation complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
