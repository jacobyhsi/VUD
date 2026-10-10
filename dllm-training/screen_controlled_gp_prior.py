"""Screen generated GP tasks and their own raw-coordinate Bayes predictions.

No training or benchmark labels are used. The posterior reference integrates
the controlled GP's lengthscale, noise regime, function and threshold with
importance sampling. It is a Monte Carlo diagnostic, not an exact oracle for
MDLM: the legacy episode loader removes a random context scale. Always report
ESS and replicate disagreement before interpreting posterior estimates.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from small_mdlm.vendor.controlled_gp_prior import (
    ControlledGPConfig, sample_at_inputs, sample_classification_datasets, sample_hyperparameters,
)


def entropy(p):
    q = p.clamp(1e-12, 1-1e-12)
    return -(q*q.log2() + (1-q)*(1-q).log2())


def posterior_summary(log_likelihood, query_p, auxiliary_p):
    """Rao--Blackwellized joint probabilities from shared prior particles.

    query_p: (particles, queries), auxiliary_p: (particles, queries, z's).
    Labels at distinct observations are independent GIVEN the latent task.
    Posterior uncertainty still creates dependence after marginalization.
    """
    if not torch.isfinite(torch.logsumexp(log_likelihood, 0)):
        raise ValueError('Reference has zero finite evidence')
    w = torch.softmax(log_likelihood, 0)
    py = (w[:, None]*query_p).sum(0)
    pu = (w[:, None, None]*auxiliary_p).sum(0)
    joint = (w[:, None, None]*query_p[:, :, None]*auxiliary_p).sum(0)
    p1 = joint / pu.clamp(min=1e-15)
    p0 = (py[:, None]-joint) / (1-pu).clamp(min=1e-15)
    p0, p1 = p0.clamp(0, 1), p1.clamp(0, 1)
    reconstructed = (1-pu)*p0 + pu*p1
    half = len(w)//2
    py_replicates = []
    for sl in (slice(0, half), slice(half, len(w))):
        ws = torch.softmax(log_likelihood[sl], 0)
        py_replicates.append((ws[:, None]*query_p[sl]).sum(0))
    effective = 1/w.square().sum()
    return {
        'particles': len(w), 'effective_sample_size': effective.item(),
        'max_particle_weight': w.max().item(),
        'replicate_p_y_max_abs_difference': (py_replicates[0]-py_replicates[1]).abs().max().item(),
        'conclusive': bool(effective >= 128 and (py_replicates[0]-py_replicates[1]).abs().max() <= .1),
        'p_y': py.tolist(), 'p_u': pu.tolist(),
        'p_y_given_u0': p0.tolist(), 'p_y_given_u1': p1.tolist(),
        'total_entropy_bits': entropy(py).tolist(),
        'expected_fixed_task_label_entropy_bits': (w[:, None]*entropy(query_p)).sum(0).tolist(),
        'marginal_consistency_max_abs_error': (py[:, None]-reconstructed).abs().max().item(),
        'log_evidence': (torch.logsumexp(log_likelihood, 0)-math.log(len(w))).item(),
    }


@torch.no_grad()
def posterior_case(d, clean, cfg, particles, chunk, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    # A small prior-generated fixture for screening; not a D=30 moons oracle.
    n, q, nz = 8, 3, 3
    context = torch.randn(1, n, d, device=device, generator=gen)
    near = context[0, 0]+.1
    middle = torch.zeros(d, device=device)
    far = torch.full((d,), 4., device=device)
    queries = torch.stack([near, middle, far])
    zs = queries[:, None] + .1*torch.randn(q, nz, d, device=device, generator=gen)
    x = torch.cat([context[0], queries, zs.flatten(0, 1)], 0)[None]
    truth_cfg = replace(cfg, clean_probability=1. if clean else 0.)
    _, labels, truth = sample_at_inputs(x, truth_cfg, gen)
    y = labels[0, :n].double()
    likelihood, ps, pus = [], [], []
    for start in range(0, particles, chunk):
        batch = min(chunk, particles-start)
        _, _, draw = sample_at_inputs(x.expand(batch, -1, -1), cfg, gen)
        standardized = ((draw['function_values'][:, :n]-draw['threshold'][:, None]) /
                        draw['noise_variance'].sqrt()[:, None])
        ll = torch.special.log_ndtr((2*y-1)[None]*standardized).sum(1)
        likelihood.append(ll.cpu())
        ps.append(draw['probabilities'][:, n:n+q].cpu())
        pus.append(draw['probabilities'][:, n+q:].view(batch, q, nz).cpu())
    result = posterior_summary(torch.cat(likelihood), torch.cat(ps), torch.cat(pus))
    return {'dimensions': d, 'context_size': n, 'generated_regime': 'clean' if clean else 'noisy',
            'scope': 'controlled GP posterior in its raw input coordinates; not an MDLM oracle',
            'query_locations': queries.cpu().tolist(), 'auxiliary_locations': zs.cpu().tolist(),
            'true_task_query_p': truth['probabilities'][0, n:n+q].cpu().tolist(), **result}


@torch.no_grad()
def generation_summary(cfg, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    hp = sample_hyperparameters(100000, cfg, gen, device)
    _, _, _, draws = sample_classification_datasets(4096, 34, 2, generator=gen,
                                                    device=device, config=cfg, return_diagnostics=True)
    h = entropy(draws['probabilities']).mean(1)
    summarize = lambda a: {'mean': a.double().mean().item(),
                          'q10_q50_q90': torch.quantile(a.double(), torch.tensor([.1,.5,.9], device=a.device,
                                                                               dtype=torch.float64)).tolist()}
    clean, noisy = draws['clean'], ~draws['clean']
    return {
        'hyperparameter_draws': 100000, 'sampled_datasets': len(h),
        'clean_fraction': hp['clean'].float().mean().item(),
        'lengthscale': summarize(hp['lengthscale']),
        'lengthscale_fraction_below_0_1': (hp['lengthscale'] < .1).float().mean().item(),
        'noise_variance_ratio_clean': summarize(hp['noise_variance'][hp['clean']]),
        'noise_variance_ratio_noisy': summarize(hp['noise_variance'][~hp['clean']]),
        'noise_variance_ratio_fraction_above_1': (hp['noise_variance'] > 1).float().mean().item(),
        'fixed_task_label_entropy_bits_clean': summarize(h[clean]),
        'fixed_task_label_entropy_bits_noisy': summarize(h[noisy]),
        'fixed_task_label_entropy_bits_all': summarize(h),
        'generator_validation_passed': bool((hp['lengthscale'] >= cfg.lengthscale_min).all()
                                            and h[clean].mean() < .05 and h[noisy].mean() > .1),
    }


@torch.no_grad()
def plot_generated_tasks(cfg, output, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.linspace(-3, 3, 101, device=device)[None, :, None].expand(4, -1, -1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for j, clean in enumerate((True, False)):
        _, _, draw = sample_at_inputs(x, replace(cfg, clean_probability=float(clean)), gen)
        for i in range(2):
            ax = axes[j, i]
            ax.plot(x[i, :, 0].cpu(), draw['probabilities'][i].cpu())
            ax.set(ylim=(-.03,1.03), xlabel='Raw input x', ylabel='Fixed-task p(y=1|x)',
                   title=f"{'Clean' if clean else 'Noisy'}: l={draw['lengthscale'][i]:.2f}, ratio={draw['noise_variance'][i]:.2g}")
    fig.savefig(output/'generated_tasks_1d.png', dpi=150)
    plt.close(fig)
    axis = torch.linspace(-3, 3, 31, device=device)
    gx, gy = torch.meshgrid(axis, axis, indexing='xy')
    points = torch.stack([gx.flatten(), gy.flatten()], -1)[None].expand(4, -1, -1)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for j, clean in enumerate((True, False)):
        _, _, draw = sample_at_inputs(points, replace(cfg, clean_probability=float(clean)), gen)
        for i in range(2):
            ax = axes[j, i]
            im = ax.imshow(draw['probabilities'][i].cpu().reshape(gx.shape), origin='lower',
                           extent=(-3,3,-3,3), cmap='RdBu_r', vmin=0, vmax=1)
            ax.set(xlabel='Raw x1', ylabel='Raw x2',
                   title=f"{'Clean' if clean else 'Noisy'}: l={draw['lengthscale'][i]:.2f}, ratio={draw['noise_variance'][i]:.2g}")
            fig.colorbar(im, ax=ax, label='Fixed-task p(y=1|x)')
    fig.savefig(output/'generated_tasks_2d.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/controlled_gp_prior.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--particles', type=int, default=131072)
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20261003)
    args = parser.parse_args()
    if args.particles < 2 or args.chunk < 1:
        raise ValueError('At least two particles and a positive chunk required')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh screening directory')
    torch.set_num_threads(2)
    cfg = ControlledGPConfig(**json.loads(args.config.read_text()))
    args.output.mkdir(parents=True)
    report = {'settings': asdict(cfg), 'seed': args.seed,
              'generation': generation_summary(cfg, args.device, args.seed), 'posterior_cases': {}}
    print(json.dumps({'generation': report['generation']}), flush=True)
    plot_generated_tasks(cfg, args.output, args.device, args.seed)
    for d in (1,2):
        for clean in (True,False):
            key = f"{d}d_{'clean' if clean else 'noisy'}"
            result = posterior_case(d, clean, cfg, args.particles, args.chunk, args.seed+10*d+int(clean), args.device)
            report['posterior_cases'][key] = result
            print(json.dumps({key: result}), flush=True)
            (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    report['scope'] = ('Synthetic prior fixtures, no benchmark labels; finite-particle raw-coordinate '
                       'posterior, not an exact MDLM posterior or a demonstration of moons performance.')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['generation']['generator_validation_passed']:
        raise RuntimeError('Generated prior failed the configured clean/smoothness sanity checks')
    print(f'Controlled GP screening complete: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
