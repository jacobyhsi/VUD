"""Compare VUD vanilla classification decompositions (MDLM vs Dream) with the true aleatoric entropy."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1

RUNS = {'MDLM-PFNClassify2D-BoTorchOS-0.8M': 'MDLM-PFNClassify2D-BoTorchOS-0.8M/va_vanilla', 'MDLM-BayesClassify2D-0.8M': 'MDLM-BayesClassify2D-0.8M/va_vanilla', 'Dream-v0-Base-7B': 'Dream-v0-Base-7B/va_vanilla'}
bits = lambda p: -(np.clip(p, 1e-12, 1) * np.log2(np.clip(p, 1e-12, 1)) + np.clip(1 - p, 1e-12, 1) * np.log2(np.clip(1 - p, 1e-12, 1)))


def decomposition(dataset, folder):
    """Per x: TU and VUD's selected Va (min Va among the 5 lowest-KL z, capped at TU); entropies in bits."""
    df = pd.read_csv(f'results/toy_classification/{dataset}/{folder}/results_test.csv')
    cols = [c for c in df.columns if c.startswith('x_x')]
    rows = []
    for key, g in df.groupby(cols, sort=True):
        tu = g['H[p(y|x,D)]'].iloc[0]
        va = min(g.nsmallest(5, 'kl_pyx_pyxz')['Va'].min(), tu)
        rows.append((*np.atleast_1d(key), tu, va, g['p(y=1|x,D)'].iloc[0]))
    return pd.DataFrame(rows, columns=cols + ['TU', 'Va', 'p1'])


for dataset in ('moons_1',):
    data = {name: decomposition(dataset, folder) for name, folder in RUNS.items()}
    ctx = pd.read_csv(f'results/toy_classification/{dataset}/{RUNS["MDLM-PFNClassify2D-BoTorchOS-0.8M"]}/D_test.csv')
    print(f'== {dataset}')
    for name, d in data.items():
        x = d[[c for c in d.columns if c.startswith('x_x')]].to_numpy()
        true_alea = bits(true_p1(dataset, x))
        print(f'  {name:26s} mean TU {d.TU.mean():.3f}  mean Va {d.Va.mean():.3f}  mean Ve {(d.TU - d.Va).mean():.3f} | '
              f'true aleatoric {true_alea.mean():.3f}  |Va-true| {np.abs(d.Va - true_alea).mean():.3f}  '
              f'Va<true on {np.mean(d.Va < true_alea - 0.05):.0%} of x')
    if dataset == 'logistic_regression':
        fig, axes = plt.subplots(1, 3, figsize=(20, 4.5), constrained_layout=True, sharey=True)
        for ax, (name, d) in zip(axes, data.items()):
            ax.plot(d.x_x1, d.TU, label='Total (VUD)')
            ax.plot(d.x_x1, d.Va, label='Aleatoric Va (VUD)')
            ax.plot(d.x_x1, d.TU - d.Va, label='Epistemic Ve (VUD)')
            ax.plot(d.x_x1, bits(true_p1(dataset, d[['x_x1']].to_numpy())), 'k--', label='True aleatoric H(p_true)')
            for v in ctx.x1:
                ax.axvline(v, color='purple', alpha=.1)
            ax.set(title=name, xlabel='x', ylabel='Entropy (bits)')
            ax.legend(fontsize=8)
    else:
        fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
        for row, (name, d) in zip(axes, data.items()):
            g1, g2 = sorted(d.x_x1.unique()), sorted(d.x_x2.unique())
            grid = lambda col: d.pivot(index='x_x2', columns='x_x1', values=col).to_numpy()
            x = d[['x_x1', 'x_x2']].to_numpy()
            d = d.assign(Ve=d.TU - d.Va, true=bits(true_p1(dataset, x)))
            panels = [('p1', 'p(y=1|x,D)', 'RdBu_r', 1), ('TU', 'Total', 'viridis', 1),
                      ('Va', 'Aleatoric Va', 'viridis', 1), ('Ve', 'Epistemic Ve', 'viridis', 1)]
            for ax, (col, title, cmap, vmax) in zip(row, panels):
                mesh = ax.pcolormesh(g1, g2, d.pivot(index='x_x2', columns='x_x1', values=col).to_numpy(),
                                     cmap=cmap, vmin=0, vmax=vmax, shading='auto')
                ax.scatter(ctx.x1, ctx.x2, c=ctx.label, cmap='RdBu_r', edgecolors='w', s=18, vmin=0, vmax=1)
                ax.contour(g1, g2, d.pivot(index='x_x2', columns='x_x1', values='true').to_numpy(),
                           levels=[0.5], colors='w', linewidths=0.8, linestyles='--')
                ax.set(title=f'{name}: {title}')
                fig.colorbar(mesh, ax=ax)
        fig.suptitle('moons_1 VUD vanilla (entropies in bits). Dashed: where the true aleatoric entropy = 0.5 bits')
    fig.savefig(f'dllm-training/tmp/logs/vud_classification_{dataset}.png', dpi=130)
print('saved')
