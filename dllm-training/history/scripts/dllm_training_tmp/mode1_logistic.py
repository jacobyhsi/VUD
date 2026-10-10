"""Mode-1 check on logistic_regression: MDLM vs the Bayes posterior of its prior, plus the truth and Dream."""
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
bayes = {os: posterior_predictive(pad(cx), cy, pad(xs[:, None]), outputscale_concentration=os, device='cuda')
         for os in (0.5, 2.0)}
for os, b in bayes.items():
    print(f'Bayes prior outputscale Gamma({os}, 0.15): effective sample size {b["effective_sample_size"]:.0f}')

def vud(folder):
    df = pd.read_csv(f'{R}/{folder}/results_test.csv')
    rows = [(x, g['H[p(y|x,D)]'].iloc[0], min(g.nsmallest(5, 'kl_pyx_pyxz')['Va'].min(), g['H[p(y|x,D)]'].iloc[0]), g['p(y=1|x,D)'].iloc[0])
            for x, g in df.groupby('x_x1')]
    return pd.DataFrame(rows, columns=['x', 'TU', 'Va', 'p1']).set_index('x').reindex(xs, method='nearest')

mdlm, dream = vud('MDLM-PFNClassify2D-0.8M/va_vanilla'), vud('Dream-v0-Base-7B/va_vanilla')
truth = bits(true_p1('logistic_regression', xs[:, None]))
inside = (xs >= cx.min()) & (xs <= cx.max())
b = bayes[0.5]
print('\nMode 1 (does the MDLM match its own prior?) -- mean abs difference, bits')
for label, mask in (('inside data', inside), ('outside data', ~inside)):
    print(f'  {label:12s} p1 {np.mean(np.abs(mdlm.p1.values - b["p1"])[mask]):.3f}   total {np.mean(np.abs(mdlm.TU.values - b["total_bits"])[mask]):.3f}   '
          f'VUD Va vs Bayes aleatoric {np.mean(np.abs(mdlm.Va.values - b["aleatoric_bits"])[mask]):.3f}')
print('\nPrior fit (Bayes aleatoric vs the true aleatoric) and the models vs truth -- mean abs difference, bits')
for label, mask in (('inside data', inside), ('outside data', ~inside)):
    print(f'  {label:12s} Bayes(PFN) {np.mean(np.abs(b["aleatoric_bits"] - truth)[mask]):.3f}  Bayes(BoTorch os) {np.mean(np.abs(bayes[2.0]["aleatoric_bits"] - truth)[mask]):.3f}  '
          f'MDLM Va {np.mean(np.abs(mdlm.Va.values - truth)[mask]):.3f}  Dream Va {np.mean(np.abs(dream.Va.values - truth)[mask]):.3f}')
print('\nFar-from-data split (outside data mean, bits): aleatoric / epistemic')
for name, a, e in (('Bayes PFN prior', b['aleatoric_bits'], b['epistemic_bits']), ('Bayes BoTorch os', bayes[2.0]['aleatoric_bits'], bayes[2.0]['epistemic_bits']),
                   ('MDLM (VUD)', mdlm.Va.values, (mdlm.TU - mdlm.Va).values), ('Dream (VUD)', dream.Va.values, (dream.TU - dream.Va).values)):
    print(f'  {name:18s} {np.mean(a[~inside]):.2f} / {np.mean(e[~inside]):.2f}')

fig, axes = plt.subplots(1, 3, figsize=(20, 4.5), constrained_layout=True, sharey=True)
for ax, (title, tot, ale) in zip(axes, (('Bayes, PFN prior Gamma(0.5, 0.15)', b['total_bits'], b['aleatoric_bits']),
                                        ('Bayes, BoTorch outputscale Gamma(2.0, 0.15)', bayes[2.0]['total_bits'], bayes[2.0]['aleatoric_bits']),
                                        ('MDLM-PFNClassify2D-0.8M (VUD)', mdlm.TU.values, mdlm.Va.values))):
    ax.plot(xs, tot, label='Total'); ax.plot(xs, ale, label='Aleatoric'); ax.plot(xs, tot - ale, label='Epistemic')
    ax.plot(xs, truth, 'k--', label='True aleatoric')
    for v in cx[:, 0]: ax.axvline(v, color='purple', alpha=.1)
    ax.set(title=title, xlabel='x', ylabel='Entropy (bits)'); ax.legend(fontsize=8)
fig.savefig('tmp/logs/mode1_logistic.png', dpi=120)
