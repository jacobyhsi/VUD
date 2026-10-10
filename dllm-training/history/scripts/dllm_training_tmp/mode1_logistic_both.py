"""Mode-1 check for both PFN-prior MDLMs on logistic_regression, each against its own prior's Bayes posterior."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from bayes_gp_classification import posterior_predictive
from evaluate_vud_classification import true_p1

R = '../results/toy_classification/logistic_regression'
bits = lambda p: -(np.clip(p,1e-12,1)*np.log2(np.clip(p,1e-12,1)) + np.clip(1-p,1e-12,1)*np.log2(np.clip(1-p,1e-12,1)))
ctx = pd.read_csv(f'{R}/MDLM-PFNClassify2D-0.8M/va_vanilla/D_test.csv')
xs = np.round(np.arange(-15, 15, 0.2), 1)
cx, cy = ctx[['x1']].to_numpy(), ctx['label'].to_numpy()
pad = lambda a: np.hstack([a, np.zeros((len(a), 1))])

def vud(folder):
    df = pd.read_csv(f'{R}/{folder}/results_test.csv')
    rows = [(x, g['H[p(y|x,D)]'].iloc[0], min(g.nsmallest(5, 'kl_pyx_pyxz')['Va'].min(), g['H[p(y|x,D)]'].iloc[0]), g['p(y=1|x,D)'].iloc[0])
            for x, g in df.groupby('x_x1')]
    return pd.DataFrame(rows, columns=['x', 'TU', 'Va', 'p1']).set_index('x').reindex(xs, method='nearest')

truth = bits(true_p1('logistic_regression', xs[:, None]))
inside = (xs >= cx.min()) & (xs <= cx.max())
models = {'PFN prior, outputscale Gamma(0.5,0.15)': ('MDLM-PFNClassify2D-0.8M/va_vanilla', 0.5),
          'BoTorch outputscale Gamma(2.0,0.15)': ('MDLM-PFNClassify2D-BoTorchOS-0.8M/va_vanilla', 2.0)}
dream = vud('Dream-v0-Base-7B/va_vanilla')
fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True, sharey=True)
for row, (name, (folder, os)) in zip(axes, models.items()):
    m, b = vud(folder), posterior_predictive(pad(cx), cy, pad(xs[:, None]), outputscale_concentration=os, device='cuda')
    print(f'== {name}')
    for label, mask in (('inside', inside), ('outside', ~inside)):
        print(f'   {label:7s} mode 1: |p1-Bayes| {np.mean(np.abs(m.p1.values-b["p1"])[mask]):.3f} |TU-Bayes total| {np.mean(np.abs(m.TU.values-b["total_bits"])[mask]):.3f} | '
              f'vs truth: Bayes aleatoric {np.mean(np.abs(b["aleatoric_bits"]-truth)[mask]):.3f}, MDLM VUD Va {np.mean(np.abs(m.Va.values-truth)[mask]):.3f} | '
              f'split ale/epi: Bayes {np.mean(b["aleatoric_bits"][mask]):.2f}/{np.mean(b["epistemic_bits"][mask]):.2f}, VUD {np.mean(m.Va.values[mask]):.2f}/{np.mean((m.TU-m.Va).values[mask]):.2f}')
    for ax, (title, tot, ale) in zip(row, ((f'Bayes, {name}', b['total_bits'], b['aleatoric_bits']), (f'MDLM VUD, {name}', m.TU.values, m.Va.values))):
        ax.plot(xs, tot, label='Total'); ax.plot(xs, ale, label='Aleatoric'); ax.plot(xs, tot - ale, label='Epistemic'); ax.plot(xs, truth, 'k--', label='True aleatoric')
        for v in cx[:, 0]: ax.axvline(v, color='purple', alpha=.1)
        ax.set(title=title, xlabel='x', ylabel='bits'); ax.legend(fontsize=7)
print(f'== Dream VUD Va vs truth: inside {np.mean(np.abs(dream.Va.values-truth)[inside]):.3f} outside {np.mean(np.abs(dream.Va.values-truth)[~inside]):.3f}')
fig.savefig('tmp/logs/mode1_logistic_both.png', dpi=110)
