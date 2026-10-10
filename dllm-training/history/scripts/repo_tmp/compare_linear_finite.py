import runpy
from pathlib import Path
import numpy as np
ns = runpy.run_path(str(Path(__file__).with_name('compare_pp_vanilla.py')))
base = ns['root'] / 'toy_regression/linear_regression/Dream-v0-Base-7B'
v = ns['read'](base / 'va_vanilla/seed_0')
p = ns['read'](base / 'va_parallel_pp/seed_0')
m = v.merge(p, on='x_x1', suffixes=('_v', '_p'))
cols = ['TU_v', 'Va_v', 'Ve_v', 'TU_p', 'Va_p', 'Ve_p']
valid = np.isfinite(m[cols]).all(axis=1)
print('INVALID', m.loc[~valid, ['x_x1','Va_v','Va_p']].to_dict('records'))
print('FINITE', int(valid.sum()), m.loc[valid,cols].mean().to_dict())
