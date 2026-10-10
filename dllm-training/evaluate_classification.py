"""Evaluate a 1-D classification checkpoint against the Bayes logistic oracle.

Suites (D=15 context, one hidden query):
  training_family: the checkpoint's own task prior.
  vud_logistic:    VUD's logistic_regression dataset, p(y=1|x) = sigmoid(0.25 x - 0.5),
                   context x ~ N(1.5, 3^2), query uniform on [-12, 12).
The model's p(y | x, D) is exact (one forward pass), so no sampling is needed.
Also plots a VUD-style sweep over x for three fixed contexts. Never trains.
"""
import argparse
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
from small_mdlm.bayes import _bernoulli_entropy, logistic_posterior_predictive
from small_mdlm.episodes import EpisodeConfig, episode
from small_mdlm.model import answer_probs

D = 15


def make_tasks(config, suite, count, seed):
    config = replace(config, context_min=D, context_max=D, max_queries=1)
    tasks = []
    for index in range(count):
        task = episode(config, seed, index, 'test')
        if suite == 'vud_logistic':
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3, index]))
            x = np.concatenate([rng.normal(1.5, 3, (D, 1)), rng.uniform(-12, 12, (1, 1))])
            p = 1 / (1 + np.exp(-(0.25 * x[:, 0] - 0.5)))
            task = {'x': x, 'p': p, 'y': (rng.random(D + 1) < p).astype(float), 'target_rows': [D]}
        tasks.append(task)
    return tasks


def model_p1(backend, xs, ys, batch_size=256):
    """p(y=1 | x_query, context) for each (context rows, query) pair; the last row is the query."""
    tok, out = backend.tokenizer, []
    for start in range(0, len(xs), batch_size):
        encoded = [tok.encode(x, np.append(y[:-1], np.nan), [len(y) - 1], hide=True)
                   for x, y in zip(xs[start:start + batch_size], ys[start:start + batch_size])]
        ids, kinds, attention, values = tok.collate(encoded, backend.device)
        probs = answer_probs(backend.model, tok, ids, kinds, attention, values).cpu()
        label_one = tok.label(1)[0]
        out += [probs[i, next(iter(e['spans'].values()))[0], label_one].item()
                for i, e in enumerate(encoded)]
    return np.array(out)


def log_loss(p1, labels):
    p = np.where(labels == 1, p1, 1 - p1)
    return float(np.mean(-np.log(np.clip(p, 1e-12, 1))))


def grid_figure(backend, prior, tasks, output):
    xs = np.arange(-12, 12, 0.2)
    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 7.5), constrained_layout=True)
    saved = {'grid_x': xs}
    for col, task in enumerate(tasks):
        cx, cy = np.round(task['x'][:-1], 1), task['y'][:-1]
        queries = [np.vstack([cx, [[x]]]) for x in xs]
        p1 = model_p1(backend, queries, [np.append(cy, np.nan)] * len(xs))
        bayes = logistic_posterior_predictive(prior, cx, cy, xs)
        top, bottom = axes[0, col], axes[1, col]
        top.plot(xs, p1, color='C0', label='MDLM p(y=1|x,D)')
        top.plot(xs, bayes['p1'], color='C2', label='Bayes p(y=1|x,D)')
        top.plot(xs, 1 / (1 + np.exp(-(0.25 * xs - 0.5))), 'k--', linewidth=1, label='True p(y=1|x)')
        top.scatter(cx[:, 0], cy, color='k', s=14, zorder=3, label='Context')
        top.set(title=f'Task {col}', xlabel='x', ylabel='p(y=1)', ylim=(-0.05, 1.05))
        bottom.plot(xs, _bernoulli_entropy(p1), color='C0', label='MDLM total entropy')
        bottom.plot(xs, bayes['total_entropy'], color='C2', label='Bayes total')
        bottom.plot(xs, bayes['aleatoric_entropy'], color='C2', linestyle=':', label='Bayes aleatoric')
        bottom.plot(xs, bayes['epistemic_entropy'], color='C2', linestyle='--', label='Bayes epistemic')
        bottom.set(xlabel='x', ylabel='Entropy (nats)', ylim=(0, 0.75))
        saved.update({f'task{col}_context_x': cx, f'task{col}_context_y': cy, f'task{col}_mdlm_p1': p1,
                      **{f'task{col}_bayes_{k}': v for k, v in bayes.items()}})
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle('VUD-style sweep on logistic_regression: fixed context (D=15), x in [-12, 12)')
    fig.savefig(output/'vud_logistic_grid.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(output/'vud_logistic_grid.npz', **saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tasks', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh evaluation directory')
    torch.set_num_threads(2)
    backend = NumericMDLM(args.checkpoint, args.device)
    prior = EpisodeConfig(**backend.config['data'])
    if not backend.tokenizer.labels or prior.input_size != 1:
        raise ValueError('This evaluation is for one-dimensional classification checkpoints')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'tasks_per_suite': args.tasks, 'context_size': D, 'seed': args.seed, 'suites': {}}
    for suite in ('training_family', 'vud_logistic'):
        tasks = make_tasks(prior, suite, args.tasks, args.seed)
        xs = [np.round(t['x'], 1) for t in tasks]
        labels = np.array([t['y'][-1] for t in tasks])
        p_true = np.array([t['p'][-1] for t in tasks])
        p_model = model_p1(backend, xs, [t['y'] for t in tasks])
        rng = np.random.default_rng(args.seed)
        shuffled = [np.append(rng.permutation(t['y'][:-1]), t['y'][-1]) for t in tasks]
        p_shuffled = model_p1(backend, xs, shuffled)
        bayes = [logistic_posterior_predictive(prior, x[:-1], t['y'][:-1], x[-1:]) for x, t in zip(xs, tasks)]
        p_bayes = np.array([b['p1'][0] for b in bayes])
        majority = np.array([np.clip(t['y'][:-1].mean(), 0.01, 0.99) for t in tasks])
        h_model, h_bayes = _bernoulli_entropy(p_model), np.array([b['total_entropy'][0] for b in bayes])
        result = {
            'log_loss': {'mdlm': log_loss(p_model, labels), 'mdlm_shuffled': log_loss(p_shuffled, labels),
                         'bayes': log_loss(p_bayes, labels), 'true_p': log_loss(p_true, labels),
                         'context_label_frequency': log_loss(majority, labels)},
            'accuracy': {'mdlm': float(np.mean((p_model > .5) == labels)),
                         'bayes': float(np.mean((p_bayes > .5) == labels))},
            'mean_abs_error_p1': {'mdlm_vs_bayes': float(np.mean(np.abs(p_model - p_bayes))),
                                  'mdlm_vs_true': float(np.mean(np.abs(p_model - p_true))),
                                  'bayes_vs_true': float(np.mean(np.abs(p_bayes - p_true)))},
            'total_entropy': {'mdlm_mean': float(h_model.mean()), 'bayes_mean': float(h_bayes.mean()),
                              'mean_abs_diff': float(np.mean(np.abs(h_model - h_bayes)))},
            'bayes_decomposition_mean': {
                'aleatoric': float(np.mean([b['aleatoric_entropy'][0] for b in bayes])),
                'epistemic': float(np.mean([b['epistemic_entropy'][0] for b in bayes]))}}
        report['suites'][suite] = result
        print(json.dumps({suite: result}), flush=True)
        if suite == 'vud_logistic':
            grid_figure(backend, prior, tasks[:3], args.output)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Evaluation complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
