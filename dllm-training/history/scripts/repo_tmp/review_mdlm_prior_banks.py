"""Read-only empirical prior diagnostics and compact saved-result comparison."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'dllm-training')
from small_mdlm.vendor import tabpfn_prior as prior

torch.set_num_threads(2)
gen = torch.Generator().manual_seed(20261003)
count = 100000
length = prior._trunc_norm_log_scaled(count, 1e-5, 10, gen, 'cpu')
scale = prior._trunc_norm_log_scaled(count, 1e-5, 10, gen, 'cpu')
logits = torch.cat([torch.ones(count, 1), prior._u(count*2, -3, 5, gen, 'cpu').reshape(count, 2)], 1)
noise = torch.tensor([1e-5, 1e-4, 1e-2])[torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1)]
print('GP_HYPERPRIORS', json.dumps({
    'lengthscale_below_0.1': float((length < .1).float().mean()),
    'lengthscale_below_0.01': float((length < .01).float().mean()),
    'observation_noise_variance_ge_function_variance': float((noise >= scale).float().mean()),
    'lengthscale_quantiles': torch.quantile(length, torch.tensor([.1,.5,.9])).tolist(),
    'noise_variance_probabilities': [float((noise == value).float().mean()) for value in [1e-5,1e-4,1e-2]],
}), flush=True)

for name in ['tabpfn_cls_3m', 'tabpfn_gpmlp50_cls_3m', 'tabpfn_gpmlp96_cls_3m']:
    root = Path('dllm-training/data_banks')/name
    rng = np.random.default_rng(20261003)
    meta = json.loads((root/'meta.json').read_text())
    ids = rng.choice(meta['datasets']-20000, 4096, replace=False)
    x,y,latent,codes,dims = [np.asarray(np.load(root/f'{key}.npy', mmap_mode='r')[ids])
                           for key in ['x','y','latent','labelling','dims']]
    p=1/(1+np.exp(-np.clip(latent.astype(float),-40,40)))
    h=-(p*np.log2(np.clip(p,1e-12,1))+(1-p)*np.log2(np.clip(1-p,1e-12,1)))
    for code in np.unique(codes):
        sel=codes==code
        a=x[sel].astype(float)
        sd=a.std(1,keepdims=True)
        z=(a-a.mean(1,keepdims=True))/np.where(sd>1e-8,sd,1)
        distance=np.linalg.norm(z[:,:,None]-z[:,None],axis=-1)
        upper=np.triu(np.ones(distance.shape[1:],bool),1)
        nearby=(distance<.2)&upper[None]
        labels=y[sel]
        disagreement=labels[:,:,None]!=labels[:,None]
        print('BANK', name, json.dumps({'member_code': int(code), 'datasets':int(sel.sum()),
              'mean_latent_label_entropy_bits':float(h[sel].mean()),
              'fraction_probabilities_0.1_to_0.9':float(((p[sel]>.1)&(p[sel]<.9)).mean()),
              'nearby_pairs':int(nearby.sum()),
              'nearby_pair_label_disagreement':float(disagreement[nearby].mean()) if nearby.any() else None}), flush=True)

records=[]
for line in Path('tmp/logs/audit_tabpfn_new_mixtures.log').read_text().splitlines():
    if not line.startswith('{'):
        continue
    record=json.loads(line)
    if record.get('kind') != 'vud_classification':
        continue
    rows=record['regions']
    records.append({'dataset':record['dataset'], 'model':record['model'],
                    'near_Va_MAE':rows['near']['Va_MAE'], 'away_Va_MAE':rows['away']['Va_MAE'],
                    'all_Va_MAE':rows['all']['Va_MAE'], 'near_Ve':rows['near']['Ve'],
                    'away_Ve':rows['away']['Ve'], 'away_TU':rows['away']['TU'],
                    'away_probability_MAE':rows['away']['p_MAE'],
                    'same_context':record['same_context']})
frame=pd.DataFrame(records)
frame.to_csv('tmp/mdlm_classification_review_20261003.csv',index=False)
print('VUD_RESULTS',flush=True)
print(frame.to_string(index=False,float_format=lambda x:f'{x:.3f}'),flush=True)
