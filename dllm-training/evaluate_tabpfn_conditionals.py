"""Held-out conditional-response evaluation, without an approximate Bayes oracle.

Uses observed labels from independent held-out latent tasks to score both the
marginal and the conditional after observing the auxiliary. NLLs are proper
predictive scores (nats), not errors against a known latent function or an
epistemic ground truth. No model weights are updated. Validation may select
checkpoints; test is forbidden for selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_vud_classification import mdlm_p1
from small_mdlm.backend import NumericMDLM


def tasks_from_bank(bank, holdout, split, count, seed):
    if not (bank/'context_size.npy').exists():
        return _tasks_from_iid_bank(bank, holdout, split, count, seed)
    arrays = {k: np.load(bank/f'{k}.npy', mmap_mode='r') for k in
              ['x', 'y', 'context_size', 'query_count', 'near_query', 'broad_query']}
    total = len(arrays['y'])
    if not (bank/'meta.json').exists() or not 2 <= holdout < total:
        raise ValueError('Need complete bank and valid holdout')
    lo, hi = (total-holdout, total-holdout//2) if split == 'validation' else (total-holdout//2, total)
    eligible = np.arange(lo, hi)[arrays['query_count'][lo:hi] == 2]
    if count > len(eligible):
        raise ValueError(f'Only {len(eligible)} two-query episodes available in {split}')
    indices = np.random.default_rng(seed).choice(eligible, count, replace=False)
    contexts, queries, auxiliaries, labels, aux_labels = [], [], [], [], []
    for i in indices:
        n = int(arrays['context_size'][i])
        x, y = np.array(arrays['x'][i, :n+2], dtype=float), np.array(arrays['y'][i, :n+2], dtype=float)
        contexts.append((x[:n], y[:n]))
        auxiliaries.append(x[n])
        queries.append(x[n+1])
        aux_labels.append(y[n])
        labels.append(y[n+1])
    return contexts, queries, auxiliaries, np.array(labels), np.array(aux_labels), indices, {
        k: np.asarray(arrays[k][indices], dtype=bool) for k in ['near_query', 'broad_query']}


def _tasks_from_iid_bank(bank, holdout, split, count, seed):
    """Legacy 34-point banks: fixed reproducible two-query held-out subsets.

    Uses the raw inputs so mdlm_p1 applies the actual backend's normalization
    both before and after adding an observed auxiliary. Splits match training.
    This is a proper label-score check, not a latent-function Bayes oracle.
    """
    if not (bank/'meta.json').exists():
        raise ValueError('Need a complete bank')
    arrays = {k: np.load(bank/f'{k}.npy', mmap_mode='r') for k in ('x', 'y')}
    total, points = arrays['y'].shape
    if not 2 <= holdout < total or points < 5:
        raise ValueError('Need a valid holdout and at least five points per dataset')
    lo, hi = (total-holdout, total-holdout//2) if split == 'validation' else (total-holdout//2, total)
    if not 1 <= count <= hi-lo:
        raise ValueError('Requested tasks must fit in the held-out split')
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(lo, hi), count, replace=False)
    contexts, queries, auxiliaries, labels, aux_labels = [], [], [], [], []
    for i in indices:
        n = int(rng.integers(3, min(32, points-2)+1))
        order = rng.permutation(points)[:n+2]
        x, y = np.array(arrays['x'][i, order], dtype=float), np.array(arrays['y'][i, order], dtype=float)
        contexts.append((x[:n], y[:n]))
        auxiliaries.append(x[n])
        queries.append(x[n+1])
        aux_labels.append(y[n])
        labels.append(y[n+1])
    return contexts, queries, auxiliaries, np.array(labels), np.array(aux_labels), indices, {}


def metrics(p, pu, p0, p1, labels, aux_labels):
    cond = np.where(aux_labels == 1, p1, p0)
    nll = lambda pred: -np.log(np.clip(np.where(labels == 1, pred, 1-pred), 1e-12, 1))
    marginal_nll, conditional_nll = nll(p), nll(cond)
    return {'marginal_nll': float(marginal_nll.mean()),
            'conditional_nll': float(conditional_nll.mean()),
            'observed_aux_gain_nats': float((marginal_nll-conditional_nll).mean()),
            'marginal_accuracy': float(((p > .5) == labels).mean()),
            'conditional_accuracy': float(((cond > .5) == labels).mean()),
            'mean_abs_conditional_response': float(abs(p1-p0).mean()),
            'marginal_consistency_MAE': float(abs(p-((1-pu)*p0+pu*p1)).mean()),
            'episodes': len(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout', type=int, default=20000)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--tasks', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--select', action='store_true')
    args = parser.parse_args()
    if args.select and args.split != 'validation':
        raise ValueError('Never select a checkpoint on the test split')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh evaluation output')
    torch.set_num_threads(2)
    contexts, queries, zs, labels, us, indices, flags = tasks_from_bank(args.bank, args.holdout, args.split, args.tasks, args.seed)
    contexts0 = [(np.vstack([x,z]), np.append(y,0)) for (x,y),z in zip(contexts,zs)]
    contexts1 = [(np.vstack([x,z]), np.append(y,1)) for (x,y),z in zip(contexts,zs)]
    args.output.mkdir(parents=True)
    report = {'bank': str(args.bank.resolve()), 'split': args.split, 'seed': args.seed,
              'bank_indices': indices.tolist(), 'checkpoints': {}}
    for checkpoint in args.checkpoint:
        backend = NumericMDLM(checkpoint, args.device)
        p, pu, p0, p1 = [mdlm_p1(backend, cx, qx, args.batch_size)
                         for cx, qx in [(contexts,queries), (contexts,zs), (contexts0,queries), (contexts1,queries)]]
        result = {'all': metrics(p,pu,p0,p1,labels,us)}
        for name, mask in flags.items():
            if mask.any():
                result[name] = metrics(*(a[mask] for a in [p,pu,p0,p1,labels,us]))
        key = str(checkpoint.resolve())
        report['checkpoints'][key] = result
        print(json.dumps({'checkpoint': key, **result}), flush=True)
        # Keep per-episode predictions for paired uncertainty/error analyses.
        np.savez_compressed(args.output/f'predictions_{len(report["checkpoints"]):03d}.npz',
                            indices=indices, p=p, pu=pu, p0=p0, p1=p1, labels=labels, auxiliary_labels=us)
        del backend
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if args.select:
        chosen = min(report['checkpoints'], key=lambda k: report['checkpoints'][k]['all']['conditional_nll'])
        report['selected_checkpoint'] = chosen
        (args.output/'selected_checkpoint.txt').write_text(chosen+'\n')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
