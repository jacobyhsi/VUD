"""Tests for the paired episode ablation; no production data required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from small_mdlm.episodes import EpisodeConfig, episode, _load_bank
from small_mdlm.tabpfn_episodes import GeometryConfig, make_paired_chunk
from small_mdlm.tokenizer import NumberTokenizer
from small_mdlm.vendor import tabpfn_prior as prior


class PairedEpisodeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.small = patch.multiple(prior, MAX_HIDDEN=16, MAX_LAYERS=3)
        self.small.start()
        self.addCleanup(self.small.stop)

    def test_identity_when_geometry_is_off(self):
        cfg = GeometryConfig(broad_query_prob=0, near_query_prob=0, draws=2)
        pair = make_paired_chunk(0, 24, 3, cfg)
        for key in pair['iid']:
            np.testing.assert_array_equal(pair['iid'][key], pair['geometry'][key])
        again = make_paired_chunk(0, 24, 3, cfg)
        for key in pair['iid']:
            np.testing.assert_array_equal(pair['iid'][key], again['iid'][key])

    def test_paired_contexts_and_close_auxiliary(self):
        cfg = GeometryConfig(broad_query_prob=1, near_query_prob=1, draws=2)
        pair = make_paired_chunk(1, 64, 9, cfg)
        a, b = pair['iid'], pair['geometry']
        self.assertEqual(set(b['query_count']), {1, 2})
        self.assertEqual(set(b['dims']), {1, 2})
        for i, (n, q) in enumerate(zip(b['context_size'], b['query_count'])):
            shared = int(n) - (q == 1)
            for key in ['x', 'y', 'latent']:
                np.testing.assert_array_equal(a[key][i, :shared], b[key][i, :shared])
            if b['dims'][i] == 1:
                self.assertTrue((b['x'][i, :, 1] == 0).all())
            if q == 2:
                # Here the full reference context is retained. With q=1 one
                # reference row is replaced, so its original std is not
                # recoverable from the stored episode (especially at n=3).
                scale = a['x'][i, :shared].std(0)
                distance = np.abs(b['x'][i, n+q-2] - b['x'][i, n+q-1]) / np.maximum(scale, 1e-8)
                self.assertLess(distance.max(), 1.2)
        self.assertTrue(np.isfinite(b['latent']).all())
        self.assertTrue(set(np.unique(b['y'])) <= {0, 1})

    def test_labels_are_generated_at_new_locations(self):
        # Known noise-free latent function makes an exact oracle for this test.
        def forward(net, x, generator=None):
            return x[..., 0]
        def threshold(net, out, generator=None):
            return torch.zeros(len(out), device=out.device)
        with patch.object(prior, 'forward', forward), patch.object(prior, 'threshold', threshold):
            pair = make_paired_chunk(4, 20, 1, GeometryConfig(draws=2))
        for bank in pair.values():
            for i, (n, q) in enumerate(zip(bank['context_size'], bank['query_count'])):
                length = int(n + q)
                np.testing.assert_array_equal(bank['y'][i, :length], bank['x'][i, :length, 0] > 0)
                np.testing.assert_array_equal(bank['latent'][i, :length] > 0, bank['y'][i, :length] > 0)

    def test_loader_splits_layout_and_no_target_leakage(self):
        pair = make_paired_chunk(2, 80, 2, GeometryConfig(draws=2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value in pair['geometry'].items():
                np.save(root/f'{key}.npy', value)
            (root/'meta.json').write_text(json.dumps({'format': 'tabpfn-fixed-episodes-v1'}))
            cfg = EpisodeConfig(task='classification_pfn', input_size=2, bank_path=directory,
                                bank_holdout=20, context_max=32, pfn_fixed_episodes=True)
            tok = NumberTokenizer(integer_digits=2, decimals=2, labels=2)
            for split, lo, hi in [('train', 0, 60), ('validation', 60, 70), ('test', 70, 80)]:
                for index in range(15):
                    e = episode(cfg, 0, index, split)
                    i = e['bank_index']
                    self.assertTrue(lo <= i < hi)
                    self.assertEqual(len(e['target_rows']), pair['geometry']['query_count'][i])
                    self.assertEqual(len(e['y']), pair['geometry']['context_size'][i] + len(e['target_rows']))
                    original = tok.encode(e['x'], e['y'], e['target_rows'], hide=True)
                    altered = e['y'].copy()
                    altered[e['target_rows']] = 1 - altered[e['target_rows']]
                    hidden = tok.encode(e['x'], altered, e['target_rows'], hide=True)
                    torch.testing.assert_close(original['ids'], hidden['ids'])
                    torch.testing.assert_close(original['values'], hidden['values'])
            _load_bank.cache_clear()

    def test_bad_config(self):
        for kwargs in [{'near_query_prob': 2}, {'draws': 0}, {'query_range': float('nan')}]:
            with self.assertRaises(ValueError):
                GeometryConfig(**kwargs)

    def test_conditional_metrics_for_coherent_toy_distribution(self):
        from evaluate_tabpfn_conditionals import metrics
        # Marginal .5; after observing u the query agrees with probability .9.
        result = metrics(np.array([.5, .5]), np.array([.5, .5]),
                         np.array([.1, .1]), np.array([.9, .9]),
                         np.array([0, 1]), np.array([0, 1]))
        self.assertAlmostEqual(result['marginal_consistency_MAE'], 0)
        self.assertAlmostEqual(result['conditional_nll'], -np.log(.9))
        self.assertGreater(result['observed_aux_gain_nats'], 0)


    def test_controlled_gp_tasks_share_arms_and_label_extra_points(self):
        from small_mdlm.vendor.controlled_gp_prior import ControlledGPConfig
        cfg = GeometryConfig(draws=2)
        pair = make_paired_chunk(5, 64, 4, cfg, gp_fraction=1.0, gp_cfg=ControlledGPConfig())
        iid, geo = pair['iid'], pair['geometry']
        for arm in (iid, geo):
            self.assertTrue(np.isfinite(arm['x']).all() and np.isfinite(arm['latent']).all())
            self.assertTrue(set(np.unique(arm['y'])) <= {0, 1})
        for row in range(64):
            obs = int(iid['context_size'][row]) - (iid['query_count'][row] == 1)
            np.testing.assert_array_equal(iid['x'][row, :obs], geo['x'][row, :obs])   # same context in both arms
            np.testing.assert_array_equal(iid['y'][row, :obs], geo['y'][row, :obs])
        # Labels follow the stored probabilities for the (nearly clean) majority of points.
        p = 1 / (1 + np.exp(-geo['latent'].astype(float)))
        used = np.arange(geo['x'].shape[1])[None] < (geo['context_size'] + geo['query_count'])[:, None]
        self.assertGreater(((p > 0.5) == (geo['y'] == 1))[used].mean(), 0.75)
        self.assertTrue(geo['broad_query'].any() and geo['near_query'].any())


    def test_structured_layouts_keep_arms_paired_and_standardized(self):
        from small_mdlm.tabpfn_episodes import _layout_points
        z = _layout_points(200, 36, 2, 1.0, torch.Generator().manual_seed(0), 'cpu')
        self.assertTrue(torch.isfinite(z).all())
        torch.testing.assert_close(z.mean(1), torch.zeros(200, 2), atol=1e-4, rtol=0)
        torch.testing.assert_close(z.std(1), torch.ones(200, 2), atol=1e-3, rtol=0)
        cfg = GeometryConfig(draws=2, layout_prob=1.0)
        for gp in (0.0, 1.0):
            pair = make_paired_chunk(6, 48, 5, cfg, gp_fraction=gp)
            iid, geo = pair['iid'], pair['geometry']
            self.assertTrue(np.isfinite(geo['x']).all() and np.isfinite(iid['latent']).all())
            for row in range(48):
                obs = int(iid['context_size'][row]) - (iid['query_count'][row] == 1)
                np.testing.assert_array_equal(iid['x'][row, :obs], geo['x'][row, :obs])


if __name__ == '__main__':
    unittest.main()
