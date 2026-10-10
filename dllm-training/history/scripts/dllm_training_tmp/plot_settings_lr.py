"""The professor's 3 settings on logistic_regression side by side: TU and AU (VUD Va) per test x, with the true
aleatoric H(true p) for reference; EU = the shaded gap. Va = min over the 5 lowest-KL z, capped at TU."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1
SETTINGS = [('Setting 1: MDLM trained on notebook-prior data', 'MDLM-TabPFNNotebook-0.8M'),
            ('Setting 2: MDLM trained on TabPFN-v1-generated data', 'MDLM-TabPFNModel-0.8M'),
            ('Setting 3: TabPFN v1 (released checkpoint)', 'TabPFN-v1')]
COL = {'tu': '#2a78d6', 'au': '#e87ba4', 'eu': '#cde2fb', 'truth': '#0b0b0b', 'y0': '#1baf7a', 'y1': '#52514e'}
bits = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
smooth = lambda x, v, bw=0.6: np.array([np.average(v, weights=np.exp(-0.5 * ((x - c) / bw) ** 2)) for c in x])
SET_COL = ['#2a78d6', '#eb6834', '#1baf7a']                                 # settings 1, 2, 3
fig, ax = plt.subplots(figsize=(11, 5.2))
for (title, name), c in zip(SETTINGS, SET_COL):
    base = f'results/toy_classification/logistic_regression/{name}/va_vanilla'
    df, ctx = pd.read_csv(f'{base}/results_test.csv'), pd.read_csv(f'{base}/D_test.csv')
    rows = []
    for x, g in df.groupby('x_x1'):
        cand = g.loc[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))]
        h = g['H[p(y|x,D)]'].iloc[0]
        rows.append((x, h, min(cand.Va.min(), h)))
    x, tu, va = np.array(rows).T
    ax.plot(x, smooth(x, tu), color=c, lw=2.2, label=f'{title}: TU')
    ax.plot(x, smooth(x, va), color=c, lw=2.2, ls='--', label='   AU (VUD Va)')
true_au = bits(np.clip(true_p1('logistic_regression', x[:, None]), 1e-9, 1 - 1e-9))
ax.plot(x, true_au, color=COL['truth'], lw=1.6, ls=':', label='True aleatoric')
for cx, cy in zip(ctx['x1'], ctx['label']):                                 # context points as a rug
    ax.plot([cx, cx], [0, 0.04], color='#0b0b0b' if cy == 1 else '#b9b8b2', lw=1.6)
ax.text(-15, 0.055, 'context points: light = y 0, dark = y 1', fontsize=8, color='#52514e')
ax.set_xlabel('x'); ax.set_ylabel('bits'); ax.set_ylim(0, 1.04)
ax.set_title('logistic_regression: TU (solid) and AU (dashed) per setting; EU = the gap between them',
             fontsize=10, loc='left')
ax.grid(axis='y', color='#e7e6e2', lw=0.6)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(ncol=2, frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13))
fig.tight_layout()
out = 'results/toy_classification/logistic_regression/settings_comparison_lr_overlay.png'
fig.savefig(out, dpi=150, facecolor='white')
print('saved', out)
