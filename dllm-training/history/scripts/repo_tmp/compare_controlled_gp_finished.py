"""Read-only comparison of saved VUD results; no model inference/training."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/home/js2723/PROJECTS/VUD')
sys.path.insert(0, str(ROOT/'dllm-training'))
from evaluate_vud_classification import true_p1

OUT = ROOT/'tmp/controlled_gp_finished_review'
OUT.mkdir(exist_ok=True)
MODELS = {
    'MDLM-TabPFN-GPMLP50-0.8M': ('Old GP50', 'tabpfn_gpmlp50_classify2d'),
    'MDLM-TabPFN-ControlledGP50-0.8M': ('Controlled GP50', 'tabpfn_controlled_gp50_classify2d'),
    'MDLM-TabPFNClassify2D-0.8M': ('Original TabPFN', 'tabpfn_classify2d'),
    'MDLM-TabPFNClassify2D-Geometry-0.8M': ('Geometry', 'tabpfn_classify2d_geometry'),
    'MDLM-TabPFNFullClassify2D-0.8M': ('Causal/non-causal MLP', 'tabpfn_full_classify2d'),
    'MDLM-BayesClassify2D-0.8M': ('BayesClassify 0.8M', 'bayes_classify2d_small'),
    'MDLM-BayesClassify2D-2.5M': ('BayesClassify 2.5M', 'bayes_classify2d'),
    'Dream-v0-Base-7B': ('Dream', None),
}
tables, report, saved = [], {}, {}
for dataset in ('logistic_regression', 'moons_1'):
    root = ROOT/'results/toy_classification'/dataset
    reference = pd.read_csv(root/'Dream-v0-Base-7B/va_vanilla/D_test.csv')
    features = [c for c in reference if c.startswith('x')]
    cx = reference[features].to_numpy(float)
    for model, (name, run) in MODELS.items():
        folder = root/model/'va_vanilla'
        if not (folder/'results_test.csv').exists():
            continue
        data = pd.read_csv(folder/'results_test.csv')
        context = pd.read_csv(folder/'D_test.csv')
        rows = []
        for key, group in data.groupby(['x_'+c for c in features]):
            candidates = group.loc[group.kl_pyx_pyxz.isin(group.kl_pyx_pyxz.nsmallest(5))]
            selected = candidates.loc[candidates.Va.idxmin()]
            h = group['H[p(y|x,D)]'].iloc[0]
            va = min(selected.Va, h) if len(features)==1 else selected.Va
            rows.append([*np.atleast_1d(key), h, va, group['p(y=1|x,D)'].iloc[0],
                         abs(selected['p(y=1|x,u=1,z,D)']-selected['p(y=1|x,u=0,z,D)']),
                         abs(selected['p(y=1|x,D)']-selected['p(y=1|x,z,D)']), len(group)])
        a = np.asarray(rows)
        x = a[:, :len(features)]
        h, va, p1, response, inconsistency, counts = a[:, len(features):].T
        truthp = np.clip(true_p1(dataset, x), 1e-12, 1-1e-12)
        truth = -truthp*np.log2(truthp)-(1-truthp)*np.log2(1-truthp)
        near = ((x[:,0]>=cx[:,0].min()) & (x[:,0]<=cx[:,0].max())) if len(features)==1 else np.linalg.norm(x[:,None]-cx[None],axis=-1).min(1)<.35
        metrics = {}
        for region, mask in [('near', near), ('away', ~near), ('all', np.ones(len(x), dtype=bool))]:
            metrics[region] = {'points': int(mask.sum()), **{k: float(v[mask].mean()) for k,v in {
                'Va_MAE': abs(va-truth), 'Va': va, 'Ve':h-va, 'TU':h, 'true_Va':truth,
                'probability_MAE':abs(p1-truthp), 'conditional_response':response,
                'marginal_inconsistency':inconsistency}.items()}}
        item = {'dataset': dataset, 'model': model, 'name': name, 'points':len(x),
                'same_context':context.equals(reference), 'z_counts': np.unique(counts).tolist(),
                'negative_epistemic_points':int((h-va < -1e-5).sum()), 'regions':metrics}
        if run:
            summary = ROOT/'dllm-training/outputs'/run/'evaluation_vud_datasets/summary.json'
            if summary.exists():
                item['predictive_40_contexts'] = json.loads(summary.read_text())['datasets'][dataset]['scores']['mdlm']
        report[f'{dataset}/{model}'] = item
        table = {'dataset':dataset, 'model':model, 'same_context':item['same_context'],
                 **{f'{r}_{k}':v for r, ms in metrics.items() for k,v in ms.items()}}
        table.update({f'predictive_{k}':v for k,v in item.get('predictive_40_contexts', {}).items()})
        tables.append(table)
        saved[(dataset,name)] = (x,h,va,truth,cx,reference.label.to_numpy())
        print(json.dumps(item), flush=True)

pd.DataFrame(tables).to_csv(OUT/'comparison.csv', index=False)
(OUT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

names = ('Old GP50','Controlled GP50','Geometry','Dream')
fig, axes = plt.subplots(len(names),3,figsize=(12,14),constrained_layout=True)
for row,name in enumerate(names):
    x,h,va,truth,cx,cy = saved[('moons_1',name)]
    xs,ys = np.unique(x[:,0]),np.unique(x[:,1])
    for col,(values,label) in enumerate(((h,'Total'),(va,'Aleatoric'),(h-va,'Epistemic'))):
        ax = axes[row,col]
        # Groupby order is x1-major; transpose to conventional x2-by-x1 image.
        mesh = ax.pcolormesh(xs,ys,values.reshape(len(xs),len(ys)).T,shading='auto',cmap='viridis',vmin=0,vmax=1)
        ax.scatter(cx[:,0],cx[:,1],c=cy,cmap='coolwarm',vmin=0,vmax=1,s=22,edgecolors='black')
        ax.set(title=f'{name}: {label}',xlabel='x1',ylabel='x2',aspect='equal')
        fig.colorbar(mesh,ax=ax,label='bits')
fig.savefig(OUT/'moons_common_scale.png',dpi=150)
plt.close(fig)
names = ('Old GP50','Controlled GP50','Causal/non-causal MLP','Dream')
fig, axes = plt.subplots(len(names),1,figsize=(10,12),sharex=True,constrained_layout=True)
for ax,name in zip(axes,names):
    x,h,va,truth,cx,cy = saved[('logistic_regression',name)]
    ax.plot(x[:,0],h,label='Total',color='C0')
    ax.plot(x[:,0],va,label='Aleatoric',color='C1')
    ax.plot(x[:,0],truth,'k--',label='True aleatoric')
    for value in cx[:,0]: ax.axvline(value,color='grey',alpha=.2,ls='--')
    ax.set(title=name,ylim=(0,1.03),ylabel='bits')
    ax.legend(loc='upper left',ncol=3,fontsize=8)
axes[-1].set_xlabel('x1')
fig.savefig(OUT/'logistic_common_scale.png',dpi=150)
plt.close(fig)
print(f'Comparison saved to {OUT}', flush=True)
