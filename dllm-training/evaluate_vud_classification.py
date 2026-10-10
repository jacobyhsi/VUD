"""Evaluate a classification checkpoint on VUD's own classification datasets.

For each dataset, draw many random contexts of VUD's size D from data.csv and
use the remaining points as queries. Compare the MDLM's exact p(y | x, D) with
baselines fitted on the same context (logistic regression, a GP classifier,
k-NN, context label frequency) and with the true p(y | x):
  logistic_regression: sigmoid(0.25 x - 0.5)                       (D = 15)
  moons_1 / moons_2:   sklearn make_moons arcs + Gaussian noise 0.1 / 0.4 (D = 30)
A shuffled-label control checks that the MDLM reads the labels. Also plots
p(y=1 | x, D) and its entropy on VUD's evaluation grids. Never trains.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from small_mdlm.backend import NumericMDLM
from small_mdlm.bayes import _bernoulli_entropy
from small_mdlm.model import answer_probs

DATA = Path(__file__).resolve().parents[1]/'datasets_toy_classification'
DATASETS = {  # name: (context size D, VUD evaluation grid per feature)
    'logistic_regression': (15, [(-12, 12, 0.2)]),
    'moons_1': (30, [(-1.5, 2.5, 0.05), (-1.5, 2.5, 0.05)]),
    'moons_2': (30, [(-3.0, 3.5, 0.05), (-2.5, 3.0, 0.05)]),
}


def load(name):
    rows = list(csv.DictReader(open(DATA/name/'data.csv')))
    columns = sorted(c for c in rows[0] if c.startswith('x'))
    return (np.array([[float(r[c]) for c in columns] for r in rows]),
            np.array([int(r['label']) for r in rows]))


def true_p1(name, x):
    if name == 'logistic_regression':
        return 1 / (1 + np.exp(-(0.25 * x[:, 0] - 0.5)))
    noise = {'moons_1': 0.1, 'moons_2': 0.4}[name]
    t = np.linspace(0, np.pi, 400)
    arcs = [np.stack([np.cos(t), np.sin(t)], 1),               # label 0 (upper)
            np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)]     # label 1 (lower)
    log_density = [np.logaddexp.reduce(-0.5 * ((x[:, None] - arc[None]) ** 2).sum(-1) / noise ** 2, 1)
                   for arc in arcs]
    return 1 / (1 + np.exp(log_density[0] - log_density[1]))


def mdlm_p1(backend, contexts, queries, batch_size=256):
    """p(y=1 | query, context) for lists of ((x_ctx, y_ctx), x_query)."""
    tok, width = backend.tokenizer, backend.config['data']['input_size']
    pad = lambda x: np.hstack([x, np.zeros((len(x), width - x.shape[1]))])
    label_one, out = tok.label(1)[0], []
    pairs = list(zip(contexts, queries))
    for start in range(0, len(pairs), batch_size):
        encoded = []
        for (cx, cy), xq in pairs[start:start + batch_size]:
            x = backend.prepare_x(pad(np.vstack([cx, xq[None]])), [len(cy)])
            encoded.append(tok.encode(x, np.append(cy, np.nan), [len(cy)], hide=True))
        ids, kinds, attention, values = tok.collate(encoded, backend.device)
        probs = answer_probs(backend.model, tok, ids, kinds, attention, values).cpu()
        out += [probs[i, next(iter(e['spans'].values()))[0], label_one].item()
                for i, e in enumerate(encoded)]
    return np.array(out)


def baselines(cx, cy, xq):
    """Classical classifiers fitted on the context only; constant if one class is present."""
    if len(np.unique(cy)) < 2:
        constant = np.full(len(xq), 0.99 if cy[0] == 1 else 0.01)
        return {'logistic_regression': constant, 'gp_classifier': constant, 'knn_5': constant}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp = GaussianProcessClassifier(ConstantKernel(1.0) * RBF(1.0), random_state=0).fit(cx, cy)
    return {'logistic_regression': LogisticRegression().fit(cx, cy).predict_proba(xq)[:, 1],
            'gp_classifier': gp.predict_proba(xq)[:, 1],
            'knn_5': KNeighborsClassifier(min(5, len(cy))).fit(cx, cy).predict_proba(xq)[:, 1]}


def scores(p1, labels, truth):
    p = np.clip(np.where(labels == 1, p1, 1 - p1), 1e-12, 1)
    return {'log_loss': float(np.mean(-np.log(p))), 'accuracy': float(np.mean((p1 > .5) == labels)),
            'mean_abs_error_vs_true_p': float(np.mean(np.abs(p1 - truth)))}


def grid_plot(backend, name, x, y, D, grid, rng, output):
    context = rng.choice(len(x), D, replace=False)
    cx, cy = x[context], y[context]
    axes_values = [np.arange(*g) for g in grid]
    if len(grid) == 1:
        xs = axes_values[0][:, None]
        p1 = mdlm_p1(backend, [(cx, cy)] * len(xs), list(xs))
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
        top.plot(xs[:, 0], p1, label='MDLM p(y=1|x,D)')
        top.plot(xs[:, 0], true_p1(name, xs), 'k--', label='True p(y=1|x)')
        top.scatter(cx[:, 0], cy, color='k', s=14, label='Context')
        top.legend(fontsize=8)
        bottom.plot(xs[:, 0], _bernoulli_entropy(p1), label='MDLM total entropy')
        bottom.set(xlabel='x1', ylabel='Entropy (nats)')
    else:
        g1, g2 = np.meshgrid(*axes_values)
        xs = np.stack([g1.ravel(), g2.ravel()], 1)
        p1 = mdlm_p1(backend, [(cx, cy)] * len(xs), list(xs)).reshape(g1.shape)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        for ax, values, title, cmap in ((axes[0], p1, 'MDLM p(y=1|x,D)', 'RdBu_r'),
                                        (axes[1], true_p1(name, xs).reshape(g1.shape), 'True p(y=1|x)', 'RdBu_r'),
                                        (axes[2], _bernoulli_entropy(p1), 'MDLM total entropy', 'viridis')):
            mesh = ax.pcolormesh(g1, g2, values, cmap=cmap, shading='auto',
                                 vmin=0, vmax=np.log(2) if 'entropy' in title else 1)
            ax.scatter(cx[:, 0], cx[:, 1], c=cy, cmap='RdBu_r', edgecolors='k', s=22, vmin=0, vmax=1)
            ax.set(title=title, xlabel='x1', ylabel='x2')
            fig.colorbar(mesh, ax=ax)
    fig.suptitle(f'{name}: one random context of D={D}')
    fig.savefig(output/f'{name}_grid.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--contexts', type=int, default=40, help='Random contexts per dataset')
    parser.add_argument('--queries', type=int, default=50, help='Held-out queries per context')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh evaluation directory')
    torch.set_num_threads(4)
    backend = NumericMDLM(args.checkpoint, args.device)
    if not backend.tokenizer.labels:
        raise ValueError('Not a classification checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'contexts_per_dataset': args.contexts, 'queries_per_context': args.queries,
              'seed': args.seed, 'datasets': {}}
    for name in args.datasets:
        D, grid = DATASETS[name]
        x, y = load(name)
        if x.shape[1] > backend.config['data']['input_size']:
            print(f'Skipping {name}: needs {x.shape[1]} inputs', flush=True)
            continue
        rng = np.random.default_rng([args.seed, len(name)])
        contexts, queries, labels, shuffled, preds = [], [], [], [], {}
        for _ in range(args.contexts):
            order = rng.permutation(len(x))
            ctx, held = order[:D], order[D:D + args.queries]
            cx, cy = x[ctx], y[ctx]
            contexts += [(cx, cy)] * len(held)
            shuffled += [(cx, rng.permutation(cy))] * len(held)
            queries += list(x[held])
            labels += list(y[held])
            for model, p in baselines(cx, cy, x[held]).items():
                preds.setdefault(model, []).extend(p)
            preds.setdefault('context_label_frequency', []).extend(
                [np.clip(cy.mean(), 0.01, 0.99)] * len(held))
        labels, xq = np.array(labels), np.array(queries)
        truth = true_p1(name, xq)
        preds = {'mdlm': mdlm_p1(backend, contexts, queries),
                 'mdlm_shuffled_labels': mdlm_p1(backend, shuffled, queries),
                 **{k: np.array(v) for k, v in preds.items()}, 'true_p': truth}
        result = {model: scores(p, labels, truth) for model, p in preds.items()}
        report['datasets'][name] = {'D': D, 'queries': int(len(labels)), 'scores': result}
        print(json.dumps({name: result}), flush=True)
        grid_plot(backend, name, x, y, D, grid, rng, args.output)
        (args.output/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Evaluation complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
