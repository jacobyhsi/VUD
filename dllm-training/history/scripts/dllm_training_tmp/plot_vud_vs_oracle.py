"""Overlay VUD vanilla decompositions (MDLM, Dream) on the Bayes oracle for linear_regression."""
import csv, glob, json, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'dllm-training')
from small_mdlm.bayes import posterior_predictive
from small_mdlm.episodes import EpisodeConfig

ROOT = 'results/toy_regression/linear_regression'
RUNS = {'MDLM-BayesLinear-1.3M': f'{ROOT}/MDLM-BayesLinear-1.3M/va_vanilla/seed_0',
        'Dream-v0-Base-7B': f'{ROOT}/Dream-v0-Base-7B/va_vanilla/seed_0'}
entropy = lambda var: 0.5 * np.log(2 * np.pi * np.e * var)


def load(folder):
    """Per x: total entropy and VUD's selected Va (min Va among the 5 lowest-KL z, capped at TU)."""
    out, groups = [], {}
    # Per-x files, or one combined results_test.csv (both layouts exist).
    paths = glob.glob(f'{folder}/results_test_x*.csv') or [f'{folder}/results_test.csv']
    for path in paths:
        for row in csv.DictReader(open(path)):
            groups.setdefault(row['x_x1'], []).append(row)
    for rows in groups.values():
        tu = float(rows[0]['H[p(y|x,D)]'])
        ranked = sorted(rows, key=lambda r: float(r['kl_pyx_pyxz']))[:5]
        va = min(min(float(r['Va']) for r in ranked), tu)
        out.append((float(rows[0]['x_x1']), tu, va, float(rows[0]['Var[y|x,D]'])))
    return np.array(sorted(out))


ctx = list(csv.DictReader(open(f"{RUNS['MDLM-BayesLinear-1.3M']}/D_test.csv")))
cx = np.array([[float(r['x1'])] for r in ctx]); cy = np.array([float(r['label']) for r in ctx])
prior = EpisodeConfig(**json.load(open('dllm-training/configs/bayes_linear_vud_small.json'))['data'])
xs = np.arange(-15, 15.001, 0.2)
bayes = posterior_predictive(prior, cx, cy, xs[:, None])


def bayes_va(points):
    """VUD's Va target under the oracle: entropy at x after adding one example (z, u) at z = x.

    The added u is set to the posterior mean; in this conjugate model the
    predictive variance depends on u only through the noise posterior."""
    out = []
    for x in points:
        u = posterior_predictive(prior, cx, cy, [[x]])['mean'][0]
        post = posterior_predictive(prior, np.vstack([cx, [[x]]]), np.append(cy, u), [[x]])
        out.append(entropy(post['total_var'][0]))
    return np.array(out)


bayes_va_grid = bayes_va(xs)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for ax, name in zip(axes, RUNS):
    data = load(RUNS[name])
    finite = np.isfinite(data).all(axis=1)
    print(f'{name}: {int((~finite).sum())} x values with non-finite entropy dropped')
    data = data[finite]
    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=.4, color='C0')
    ax.scatter(data[:, 0], data[:, 2], s=10, alpha=.4, color='C3')
    k = 7  # moving average for readability
    smooth = lambda v: np.convolve(v, np.ones(k) / k, mode='same')
    ax.plot(data[k:-k, 0], smooth(data[:, 1])[k:-k], color='C0', label=f'{name} total (VUD)')
    ax.plot(data[k:-k, 0], smooth(data[:, 2])[k:-k], color='C3', label=f'{name} aleatoric Va (VUD)')
    ax.plot(xs, entropy(bayes['total_var']), 'k-', linewidth=2, label='Bayes total')
    ax.plot(xs, entropy(bayes['aleatoric_var']), 'k:', linewidth=2, label='Bayes aleatoric')
    ax.plot(xs, bayes_va_grid, color='C3', linestyle='--', linewidth=2, label='Bayes Va (VUD target)')
    ax.axhline(entropy(4.0), color='gray', linestyle='--', linewidth=1, label='True noise (sigma=2)')
    for x in cx[:, 0]:
        ax.axvline(x, color='purple', alpha=.08)
    ax.set(title=f'{name}: {len(data)} x values', xlabel='x', ylabel='Entropy (nats)', ylim=(0.8, 4.2))
    ax.legend(fontsize=8)
    inside = (data[:, 0] >= -2) & (data[:, 0] <= 6)
    b = posterior_predictive(prior, cx, cy, data[:, :1])
    for label, mask in (('near data [-2,6]', inside), ('far |x| outside', ~inside)):
        target_va = bayes_va(data[mask, 0])
        print(f'{name:22s} {label:18s} TU err={np.mean(data[mask,1]-entropy(b["total_var"][mask])):+.2f} '
              f'Va err vs Bayes Va={np.mean(data[mask,2]-target_va):+.2f} '
              f'Ve(VUD)={np.mean(data[mask,1]-data[mask,2]):.2f} Ve(Bayes)={np.mean(entropy(b["total_var"][mask])-entropy(b["aleatoric_var"][mask])):.2f}')
fig.savefig('dllm-training/tmp/logs/vud_vs_oracle_linear_vanilla.png', dpi=150)
print('saved')
