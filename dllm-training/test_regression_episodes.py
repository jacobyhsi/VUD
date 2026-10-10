import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from small_mdlm.episodes import EpisodeConfig, episode, normalize_targets
from small_mdlm.regression_episodes import make_regression_chunk
from small_mdlm.tabpfn_episodes import GeometryConfig
from small_mdlm.tokenizer import NumberTokenizer
from small_mdlm.vendor.regression_priors import HetGPConfig, het_gp_at


class RegressionEpisodeTests(unittest.TestCase):
    def test_het_gp_noise_regimes_and_heteroskedasticity(self):
        x = torch.randn(400, 30, 1, dtype=torch.float64)
        y, f, s = het_gp_at(x, HetGPConfig(), torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(y).all() and (s > 0).all())
        ratio = s.max(1).values / s.min(1).values
        self.assertGreater((ratio > 2).float().mean(), 0.2)                     # some tasks vary their noise
        self.assertGreater((ratio < 1 + 1e-9).float().mean(), 0.3)              # some are homoskedastic
        self.assertLess(s.median(1).values.min(), 0.05)                         # clean tasks exist

    def test_chunk_layout_and_reproducibility(self):
        cfg = GeometryConfig(draws=2, layout_prob=0.75)
        a = make_regression_chunk(1, 64, 3, cfg)
        b = make_regression_chunk(1, 64, 3, cfg)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertTrue(set(np.unique(a['gp_task'])) == {0, 1})
        for row in range(64):
            used = int(a['context_size'][row] + a['query_count'][row])
            self.assertTrue(np.isfinite(a['y'][row, :used]).all() and (a['std'][row, :used] >= 0).all())
            self.assertTrue(np.all(a['x'][row, used:] == 0))                     # padding untouched

    def test_episodes_normalise_x_and_y_by_the_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_regression_chunk(0, 300, 0, GeometryConfig(draws=2))
            for k, v in chunk.items():
                np.save(root/f'{k}.npy', v)
            (root/'meta.json').write_text(json.dumps({'format': 'regression-fixed-episodes-v1'}))
            cfg = EpisodeConfig(task='regression_pfn', input_size=1, bank_path=str(root), bank_holdout=60,
                                context_min=3, context_max=32, max_queries=2, pfn_fixed_episodes=True)
            tok = NumberTokenizer(integer_digits=2, decimals=3)
            for index in range(40):
                e = episode(cfg, 0, index)
                observed = [r for r in range(len(e['y'])) if r not in e['target_rows']]
                self.assertAlmostEqual(e['y'][observed].mean(), 0, places=6)
                self.assertTrue(abs(e['y'][observed].std() - 1) < 1e-6 or e['y'][observed].std() < 1e-6)
                self.assertTrue(np.all(np.abs(e['y']) <= 50))
                tok.encode(e['x'], e['y'], e['target_rows'])                     # fits the tokenizer range
            self.assertIn(episode(cfg, 0, 0, 'test')['bank_index'], range(270, 300))

    def test_normalize_targets_round_trip(self):
        y = np.array([1.0, 3.0, 5.0, np.nan])
        z, m, s = normalize_targets(np.nan_to_num(y), [0, 1, 2])
        np.testing.assert_allclose(z[:3] * s + m, y[:3])


if __name__ == '__main__':
    unittest.main()
