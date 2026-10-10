"""Evaluate fixed MDLM weights against least squares and shuffled-label controls.

Uses independent test episodes, D=15, a single hidden query, and multiple
samples per query. Does not train or modify the checkpoint.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from small_mdlm.backend import NumericMDLM
from small_mdlm.bayes import posterior_predictive
from small_mdlm.episodes import EpisodeConfig, episode
from small_mdlm.model import sample


def make_tasks(config, suite, count, seed):
    config = replace(config, context_min=15, context_max=15, max_queries=1)
    tasks = []
    for index in range(count):
        task = episode(config, seed, index, 'test')
        if suite == 'vud_linear':
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3, index]))
            task['x'] = np.concatenate([rng.normal(1, 2, (15, 1)),
                                        rng.uniform(-15, 15, (1, 1))])
            task['weights'] = np.array([-1.0])
            task['bias'] = 3.0
            task['noise_std'] = 2.0
            task['mean'] = -task['x'][:, 0] + 3
            task['y'] = -task['x'][:, 0] + 3 + rng.normal(0, 2, 16)
        tasks.append(task)
    return tasks


def contexts(tasks, tokenizer, prior, seed, shuffle=False):
    examples, baselines = [], []
    for index, task in enumerate(tasks):
        x, y = task['x'], task['y'].copy()
        if shuffle:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 4, index]))
            y[:-1] = rng.permutation(y[:-1])
        # Give baselines exactly the same rounded inputs/observations as MDLM.
        rounded_x = np.round(x, tokenizer.decimals)
        observed_y = np.round(y[:-1], tokenizer.decimals)
        design = np.column_stack([rounded_x[:-1], np.ones(len(x) - 1)])
        weights = np.linalg.lstsq(design, observed_y, rcond=None)[0]
        ols = np.append(rounded_x[-1], 1) @ weights
        # Exact posterior predictive under the checkpoint's training prior.
        bayes = posterior_predictive(prior, rounded_x[:-1], observed_y, rounded_x[-1:])
        baselines.append((float(ols), float(observed_y.mean()), float(bayes['mean'][0]),
                          float(bayes['total_var'][0]), float(bayes['aleatoric_var'][0]),
                          float(bayes['epistemic_var'][0])))
        # Hidden label must not reach tokenization; score against it separately.
        y[-1] = np.nan
        examples.append(tokenizer.encode(x, y, [len(y) - 1], hide=True))
    return examples, np.asarray(baselines)


@torch.no_grad()
def predictions(backend, examples, draws, batch_size, seed):
    # Ordering, mask patterns and sampling seeds are identical across controls.
    flattened = [e for e in examples for _ in range(draws)]
    output = []
    for start in range(0, len(flattened), batch_size):
        batch = flattened[start:start + batch_size]
        ids, kinds, attention, values = backend.tokenizer.collate(batch, backend.device)
        generated = sample(backend.model, backend.tokenizer, ids, kinds, attention,
                           seed=seed + start, steps=backend.config['training']['sample_steps'], values=values)
        generated = generated.cpu()
        for row, example in enumerate(batch):
            positions = next(iter(example['spans'].values()))
            output.append(backend.tokenizer.decode_number(generated[row, positions]))
    return np.asarray(output).reshape(len(examples), draws)


def true_mean(task):
    """Noiseless E[y | x] at the query (general-regression tasks store it per row)."""
    if 'mean' in task:
        return float(task['mean'][-1])
    return float(task['x'][-1] @ task['weights'] + task['bias'])


def true_noise_std(task):
    return float(task['std'][-1]) if 'std' in task else float(task['noise_std'])


def calibration(draws, noisy, baseline, tasks):
    """Compare the MDLM's sample spread with the Bayes posterior predictive."""
    mdlm_var = draws.var(axis=1, ddof=1)
    bayes_mean, bayes_var = baseline[:, 2], baseline[:, 3]
    low, high = np.quantile(draws, [0.05, 0.95], axis=1)
    half = 1.6449 * np.sqrt(bayes_var)
    return {
        'mdlm_mean_vs_bayes_mean_mse': float(np.mean((draws.mean(axis=1) - bayes_mean) ** 2)),
        'mdlm_predictive_var_mean': float(mdlm_var.mean()),
        'bayes_predictive_var_mean': float(bayes_var.mean()),
        'bayes_aleatoric_var_mean': float(baseline[:, 4].mean()),
        'bayes_epistemic_var_mean': float(baseline[:, 5].mean()),
        'true_noise_var_mean': float(np.mean([true_noise_std(t) ** 2 for t in tasks])),
        'median_var_ratio_mdlm_over_bayes': float(np.median(mdlm_var / bayes_var)),
        'coverage90_mdlm_empirical_quantiles': float(np.mean((noisy >= low) & (noisy <= high))),
        'coverage90_bayes_gaussian_approx': float(np.mean(np.abs(noisy - bayes_mean) <= half))}


def grid_figure(backend, tasks, prior, draws, batch_size, seed, output):
    """VUD-style sweep: fixed contexts, queries on a grid, MDLM vs the Bayes oracle."""
    xs = np.arange(-15, 15.001, 1.0)
    tok = backend.tokenizer
    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 8), constrained_layout=True)
    saved = {'grid_x': xs}
    for col, task in enumerate(tasks):
        x_ctx, y_ctx = np.round(task['x'][:-1], tok.decimals), np.round(task['y'][:-1], tok.decimals)
        examples = [tok.encode(np.vstack([x_ctx, [[x]]]), np.append(y_ctx, np.nan),
                               [len(y_ctx)], hide=True) for x in xs]
        samples = predictions(backend, examples, draws, batch_size, seed + 20000 + col)
        bayes = posterior_predictive(prior, x_ctx, y_ctx, xs[:, None])
        mean, var = samples.mean(axis=1), samples.var(axis=1, ddof=1)
        top, bottom = axes[0, col], axes[1, col]
        top.fill_between(xs, bayes['mean'] - 2 * np.sqrt(bayes['total_var']),
                         bayes['mean'] + 2 * np.sqrt(bayes['total_var']), color='C2', alpha=.2,
                         label='Bayes ±2 sd')
        top.plot(xs, bayes['mean'], color='C2', label='Bayes mean')
        top.fill_between(xs, mean - 2 * np.sqrt(var), mean + 2 * np.sqrt(var), color='C0',
                         alpha=.2, label='MDLM ±2 sd')
        top.plot(xs, mean, color='C0', label='MDLM mean')
        top.plot(xs, xs * task['weights'][0] + task['bias'], 'k--', linewidth=1, label='True line')
        top.scatter(x_ctx[:, 0], y_ctx, color='k', s=12, zorder=3, label='Context')
        top.set(title=f'Task {col}', xlabel='x', ylabel='y')
        bottom.plot(xs, var, color='C0', label='MDLM variance')
        bottom.plot(xs, bayes['total_var'], color='C2', label='Bayes total')
        bottom.plot(xs, bayes['aleatoric_var'], color='C2', linestyle=':', label='Bayes aleatoric')
        bottom.plot(xs, bayes['epistemic_var'], color='C2', linestyle='--', label='Bayes epistemic')
        bottom.set(xlabel='x', ylabel='Predictive variance', yscale='log')
        saved.update({f'task{col}_samples': samples, f'task{col}_context_x': x_ctx,
                      f'task{col}_context_y': y_ctx, f'task{col}_bayes_mean': bayes['mean'],
                      f'task{col}_bayes_total_var': bayes['total_var'],
                      f'task{col}_bayes_aleatoric_var': bayes['aleatoric_var']})
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle('VUD-style grid: fixed context (D=15), queries on x in [-15, 15]')
    fig.savefig(output/'vud_linear_grid.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(output/'vud_linear_grid.npz', **saved)


def ci_mean(values, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def score(pred, truth, noisy):
    error = (pred - truth) ** 2
    return {'latent_mse': float(error.mean()),
            'latent_mse_ci95': ci_mean(error, 71),
            'noisy_target_mse': float(np.mean((pred - noisy) ** 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tasks', type=int, default=256)
    parser.add_argument('--draws', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if min(args.tasks, args.draws, args.batch_size) < 2:
        raise ValueError('Use at least two tasks, draws, and batch rows')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh evaluation directory')
    torch.set_num_threads(2)
    backend = NumericMDLM(args.checkpoint, args.device)
    if backend.config['data']['input_size'] != 1:
        raise ValueError('This evaluation is for one-dimensional checkpoints')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'tasks_per_suite': args.tasks, 'draws_per_query': args.draws,
              'context_size': 15, 'seed': args.seed, 'temperature': 1.0,
              'sample_steps': backend.config['training']['sample_steps'],
              'notes': ['Single-query evaluation; not joint-distribution calibration.',
                        *(['Bayes oracle uses only the linear part of the prior; it is not exact '
                           'for this general-regression checkpoint.']
                          if backend.config['data'].get('task') == 'regression_mix' else []),
                        'MDLM mean uses finite Monte Carlo samples; OLS is deterministic.',
                        'Confidence intervals bootstrap tasks, not individual draws.',
                        'VUD suite resamples contexts per task; it is not the fixed-context VUD grid.'],
              'suites': {}}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for col, suite in enumerate(('training_family', 'vud_linear')):
        prior = EpisodeConfig(**backend.config['data'])
        tasks = make_tasks(prior, suite, args.tasks, args.seed)
        truth = np.array([true_mean(t) for t in tasks])
        noisy = np.array([t['y'][-1] for t in tasks])
        examples, baseline = contexts(tasks, backend.tokenizer, prior, args.seed)
        shuffled, corrupted_baseline = contexts(tasks, backend.tokenizer, prior, args.seed, shuffle=True)
        print(f'{suite}: sampling correct contexts', flush=True)
        draws = predictions(backend, examples, args.draws, args.batch_size, args.seed + 10000)
        print(f'{suite}: sampling shuffled-label contexts', flush=True)
        shuffled_draws = predictions(backend, shuffled, args.draws, args.batch_size, args.seed + 10000)
        mean, shuffled_mean = draws.mean(axis=1), shuffled_draws.mean(axis=1)
        models = {'mdlm_mean': mean, 'mdlm_shuffled_mean': shuffled_mean,
                  'least_squares': baseline[:, 0], 'least_squares_shuffled': corrupted_baseline[:, 0],
                  'context_mean': baseline[:, 1], 'bayes_mean': baseline[:, 2],
                  'zero': np.zeros(len(tasks)), 'oracle': truth}
        result = {name: score(pred, truth, noisy) for name, pred in models.items()}
        degradation = (shuffled_mean - truth) ** 2 - (mean - truth) ** 2
        result['shuffle_effect'] = {
            'latent_mse_increase': float(degradation.mean()),
            'paired_ci95': ci_mean(degradation, args.seed),
            'rms_change_in_mean_prediction': float(np.sqrt(np.mean((mean - shuffled_mean) ** 2))),
            'fraction_changed_draws_with_paired_seeds': float(np.mean(draws != shuffled_draws))}
        result['sampling'] = {
            'average_single_draw_noisy_mse': float(np.mean((draws - noisy[:, None]) ** 2)),
            'mean_predictive_variance': float(draws.var(axis=1, ddof=1).mean()),
            'mean_mc_standard_error': float(np.mean(draws.std(axis=1, ddof=1) / np.sqrt(args.draws)))}
        result['calibration'] = calibration(draws, noisy, baseline, tasks)
        if suite == 'vud_linear':
            grid_figure(backend, tasks[:3], prior, args.draws, args.batch_size, args.seed, args.output)
        report['suites'][suite] = result
        np.savez_compressed(args.output/f'{suite}_samples.npz', samples=draws,
                            shuffled_samples=shuffled_draws, latent_truth=truth, noisy_target=noisy)
        with (args.output/f'{suite}_predictions.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['task', 'query_x', 'family', 'weight', 'bias', 'latent_truth', 'noisy_target', *models])
            for i, task in enumerate(tasks):
                # General-regression tasks have no single line; leave weight/bias empty.
                weight = task['weights'][0] if 'weights' in task else ''
                writer.writerow([i, task['x'][-1, 0], task.get('family', 'linear'), weight,
                                 task.get('bias', ''), truth[i], noisy[i],
                                 *[pred[i] for pred in models.values()]])
        print(json.dumps({suite: result}), flush=True)
        ax = axes[0, col]
        for name, label, color in (('bayes_mean', 'Bayes mean', 'C2'),
                                   ('mdlm_mean', 'MDLM mean', 'C0'),
                                   ('mdlm_shuffled_mean', 'MDLM shuffled', 'C1')):
            ax.scatter(truth, models[name], s=14, alpha=.55, label=label, color=color)
        limits = [min(truth.min(), *(p.min() for p in (mean, shuffled_mean, baseline[:, 2]))),
                  max(truth.max(), *(p.max() for p in (mean, shuffled_mean, baseline[:, 2])))]
        ax.plot(limits, limits, 'k--', linewidth=1)
        ax.set(xlabel='True noiseless query output', ylabel='Predicted mean',
               title=suite.replace('_', ' ').title())
        ax.legend(fontsize=8)
        names = ['mdlm_mean', 'mdlm_shuffled_mean', 'bayes_mean', 'least_squares', 'zero']
        values = [result[name]['latent_mse'] for name in names]
        axes[1, col].bar(range(len(names)), values, color=['C0', 'C1', 'C2', 'C3', 'gray'])
        axes[1, col].set_xticks(range(len(names)), ['MDLM', 'Shuffled\nMDLM', 'Bayes', 'Least\nsquares', 'Zero'])
        axes[1, col].set(ylabel='MSE against noiseless query output')
        for i, value in enumerate(values):
            axes[1, col].annotate(f'{value:.2f}', (i, value), xytext=(0, 4),
                                   textcoords='offset points', ha='center', fontsize=9)
        axes[1, col].margins(y=.15)
        (args.output/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    fig.suptitle(f'Held-out regression: D=15, {args.tasks} tasks/suite, {args.draws} MDLM samples/query')
    fig.savefig(args.output/'comparison.png', dpi=180)
    plt.close(fig)
    print(f'Evaluation complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
