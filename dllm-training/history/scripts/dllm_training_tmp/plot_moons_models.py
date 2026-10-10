"""Total / aleatoric (VUD Va) / epistemic (Ve) maps on moons_1 for several models, one row each.
Usage (from the repo root): python dllm-training/tmp/scripts/plot_moons_models.py <out.png> <model> [<model> ...]"""
import sys
import pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = 'results/toy_classification/moons_1'
out, models = sys.argv[1], sys.argv[2:]
fig, axes = plt.subplots(len(models), 3, figsize=(13, 4 * len(models)), constrained_layout=True, squeeze=False)
for row, m in zip(axes, models):
    df = pd.read_csv(f'{R}/{m}/va_vanilla/results_test.csv')
    rows = []
    for (x1, x2), s in df.groupby(['x_x1', 'x_x2']):
        tu = s['H[p(y|x,D)]'].iloc[0]
        rows.append((x1, x2, tu, min(s.nsmallest(5, 'kl_pyx_pyxz')['Va'].min(), tu)))
    d = pd.DataFrame(rows, columns=['x1', 'x2', 'TU', 'Va']).assign(Ve=lambda t: t.TU - t.Va)
    ctx = pd.read_csv(f'{R}/{m}/va_vanilla/D_test.csv')
    for ax, col in zip(row, ['TU', 'Va', 'Ve']):
        piv = d.pivot(index='x2', columns='x1', values=col)
        mesh = ax.pcolormesh(piv.columns, piv.index, piv.to_numpy(), vmin=0, vmax=1, cmap='viridis', shading='auto')
        ax.scatter(ctx.x1, ctx.x2, c=ctx.label, cmap='coolwarm', edgecolors='w', s=18)
        ax.set_title(f'{m}: {col}', fontsize=9)
    fig.colorbar(mesh, ax=row)
fig.savefig(out, dpi=70)
