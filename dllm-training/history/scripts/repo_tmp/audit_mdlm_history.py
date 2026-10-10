"""Read-only audit of saved MDLM experiments; no model inference or training."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, 'dllm-training')
from evaluate_vud_classification import true_p1

def emit(kind, **data):
    print(json.dumps({'kind': kind, **data}), flush=True)

for run in sorted(Path('dllm-training/outputs').iterdir()):
    metric_file = run / 'metrics.jsonl'
    if metric_file.exists():
        records = [json.loads(s) for s in metric_file.read_text().splitlines() if s.strip()]
        if records:
            last = records[-1]
            emit('training', run=run.name, final=last,
                 at40k=next((r for r in records if r['step'] == 40000), None))
    for p in sorted(run.glob('evaluation*/summary.json')):
        report = json.loads(p.read_text())
        if 'datasets' in report:
            emit('classification_eval', run=run.name, evaluation=p.parent.name,
                 datasets={k: v['scores'] for k, v in report['datasets'].items()})
        elif 'suites' in report:
            selected = {}
            for key, value in report['suites'].items():
                selected[key] = {k: v for k, v in value.items()
                                 if k in ['mdlm', 'mdlm_mean', 'bayes_mean', 'least_squares', 'mdlm_predictive_var_mean', 'true_noise_var_mean']}
            emit('regression_eval', run=run.name, evaluation=p.parent.name, suites=selected)

for dataset in ['logistic_regression', 'moons_1']:
    root = Path('results/toy_classification') / dataset
    reference = pd.read_csv(root / 'Dream-v0-Base-7B/va_vanilla/D_test.csv')
    for folder in sorted(root.glob('*/va_vanilla')):
        if not (folder.parent.name.startswith('MDLM') or folder.parent.name == 'Dream-v0-Base-7B'):
            continue
        if not (folder / 'results_test.csv').exists():
            continue
        df = pd.read_csv(folder / 'results_test.csv')
        context = pd.read_csv(folder / 'D_test.csv')
        cols = [c for c in context if c.startswith('x')]
        cx = context[cols].to_numpy(float)
        rows = []
        for key, g in df.groupby(['x_' + c for c in cols]):
            h = g['H[p(y|x,D)]'].iloc[0]
            kl = g.kl_pyx_pyxz
            candidates = g.loc[kl.isin(kl.nsmallest(5))]
            selected = candidates.loc[candidates.Va.idxmin()]
            va = selected.Va
            # Match the 1-D plot's clipping and the 2-D plot's unclipped selection.
            if len(cols) == 1:
                va = min(va, h)
            rows.append((*np.atleast_1d(key), h, va,
                         g['p(y=1|x,D)'].iloc[0], selected.kl_pyx_pyxz,
                         abs(selected['p(y=1|x,D)']-selected['p(y=1|x,z,D)']),
                         abs(selected['p(y=1|x,u=1,z,D)']-selected['p(y=1|x,u=0,z,D)']), len(g)))
        a = np.array(rows)
        x = a[:, :len(cols)]
        h, va, p1, kl, dp, response, counts = a[:, len(cols):].T
        truthp = np.clip(true_p1(dataset, x), 1e-12, 1-1e-12)
        truth = -truthp*np.log2(truthp)-(1-truthp)*np.log2(1-truthp)
        near = ((x[:,0] >= cx.min()) & (x[:,0] <= cx.max())) if len(cols)==1 else np.linalg.norm(x[:,None]-cx[None],axis=-1).min(1)<.35
        regions = {}
        for name, mask in [('near', near), ('away', ~near), ('all', np.ones(len(x), dtype=bool))]:
            regions[name] = {k: float(v[mask].mean()) for k,v in {
                'Va_MAE': abs(va-truth), 'Va':va, 'Ve':h-va, 'TU':h,
                'p_MAE': abs(p1-truthp), 'selected_KL':kl,
                'marginal_inconsistency':dp, 'conditional_response':response,
                'true_Va':truth}.items()}
        emit('vud_classification', dataset=dataset, model=folder.parent.name,
             points=len(a), z_counts=np.unique(counts).tolist(),
             same_context=context.equals(reference), negative_Ve=int(np.sum(h-va < -1e-5)), regions=regions)
        if 'TabPFN' in folder.parent.name:
            filename = 'mode1_bag_moons_1_tabpfn2500.csv' if dataset == 'moons_1' else 'mode1_bag_logistic_regression.csv'
            ref = pd.read_csv(Path('dllm-training/tmp/logs') / filename)
            ref = ref[ref.bag == 'TabPFN alone'].sort_values(cols)
            if len(ref) == len(x) and np.allclose(ref[cols].to_numpy(), x):
                emit('approx_bayes_comparison', dataset=dataset, model=folder.parent.name,
                     p_MAE_near=float(abs(p1[near]-ref.p1.to_numpy()[near]).mean()),
                     p_MAE_away=float(abs(p1[~near]-ref.p1.to_numpy()[~near]).mean()),
                     TU_bias_near=float((h[near]-ref.TU.to_numpy()[near]).mean()),
                     TU_bias_away=float((h[~near]-ref.TU.to_numpy()[~near]).mean()))

for dataset in ['gaps', 'linear_regression']:
    root = Path('results/toy_regression') / dataset
    for path in sorted(root.glob('MDLM*/va_vanilla/seed_0/results_test.csv')):
        df = pd.read_csv(path)
        context = pd.read_csv(path.parent/'D_test.csv')
        rows = []
        feature = next(c for c in context if c.startswith('x'))
        for x,g in df.groupby('x_' + feature):
            valid = g[g.kl_pyx_pyxz.isin(g.kl_pyx_pyxz.nsmallest(5))]
            rows.append([x,g['H[p(y|x,D)]'].iloc[0],valid.Va.min(),g['p(y|x,D)_mean'].iloc[0], g['p(y|x,D)_std'].iloc[0]])
        a=np.array(rows)
        emit('vud_regression', dataset=dataset, model=path.parents[2].name,
             points=len(a), context_size=len(context), context_x=context[feature].tolist(),
             entropy_range=[float(a[:,1].min()),float(a[:,1].max())],
             Va_range=[float(a[:,2].min()),float(a[:,2].max())],
             negative_Ve=int(np.sum(a[:,1]<a[:,2])),
             samples=a[::10].round(4).tolist())

print('COMPACT TRAINING HISTORY', flush=True)
for path in sorted(Path('dllm-training/outputs').glob('*/metrics.jsonl')):
    r=json.loads(path.read_text().splitlines()[-1])
    fields=['step','validation_mse','shuffled_mse','mse','mae','validation_loss','label_log_loss','accuracy','abs_error_vs_true_p']
    print(path.parent.name, {k:round(r[k],5) for k in fields if k in r}, flush=True)
print('COMPACT CLASSIFICATION EVALUATIONS', flush=True)
for path in sorted(Path('dllm-training/outputs').glob('*/evaluation_vud_datasets/summary.json')):
    r=json.loads(path.read_text())
    print(path.parents[1].name, {k:v['scores']['mdlm'] for k,v in r['datasets'].items()}, flush=True)
print('COMPACT LINEAR EVALUATIONS', flush=True)
for path in sorted(Path('dllm-training/outputs').glob('*/evaluation_seed2026/summary.json')):
    r=json.loads(path.read_text())
    print(path.parents[1].name, {k:v.get('mdlm_mean', v.get('mdlm', {})) for k,v in r.get('suites',{}).items()}, flush=True)
