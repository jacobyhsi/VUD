import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from small_mdlm.vendor.controlled_gp_prior import (
    ControlledGPConfig, sample_at_inputs, sample_classification_datasets, sample_hyperparameters,
)


class ControlledGPPriorTests(unittest.TestCase):
    def generator(self, seed=0):
        return torch.Generator().manual_seed(seed)

    def test_invalid_settings(self):
        for override in ({'lengthscale_min': 0}, {'lengthscale_max': 0.1},
                         {'clean_probability': -0.1}, {'clean_probability': math.nan},
                         {'clean_variance_ratio_max': math.inf}, {'noisy_variance_ratio_min': -1},
                         {'threshold_std': -1}, {'jitter': 0}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                ControlledGPConfig(**override)

    def test_hyperparameter_ranges_and_regime_balance(self):
        cfg = ControlledGPConfig()
        hp = sample_hyperparameters(10000, cfg, self.generator())
        self.assertTrue(((hp['lengthscale'] >= cfg.lengthscale_min) &
                         (hp['lengthscale'] <= cfg.lengthscale_max)).all())
        self.assertAlmostEqual(hp['clean'].float().mean().item(), .5, delta=.02)
        for mask, lo, hi in ((hp['clean'], cfg.clean_variance_ratio_min, cfg.clean_variance_ratio_max),
                             (~hp['clean'], cfg.noisy_variance_ratio_min, cfg.noisy_variance_ratio_max)):
            self.assertTrue(((hp['noise_variance'][mask] >= lo) & (hp['noise_variance'][mask] <= hi)).all())

    def test_shapes_reproducibility_and_finite_outputs(self):
        for d in (1, 2):
            result = sample_classification_datasets(32, 34, d, generator=self.generator(8), return_diagnostics=True)
            again = sample_classification_datasets(32, 34, d, generator=self.generator(8), return_diagnostics=True)
            x, latent, y, hp = result
            self.assertEqual(x.shape, (32, 34, d))
            self.assertTrue(torch.isfinite(x).all() and torch.isfinite(latent).all())
            self.assertTrue(set(y.unique().tolist()) <= {0., 1.})
            for first, second in zip(result[:3], again[:3]):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            expected = torch.special.ndtr((hp['function_values'] - hp['threshold'][:, None]) /
                                          hp['noise_variance'].sqrt()[:, None]).clamp(1e-6, 1-1e-6)
            torch.testing.assert_close(latent.double().sigmoid(), expected, rtol=1e-6, atol=1e-8)

    def test_nearby_locations_share_a_latent_function(self):
        x = torch.tensor([[[0., 0.], [1e-4, 0.]]]).expand(512, -1, -1)
        _, _, hp = sample_at_inputs(x, generator=self.generator(2))
        difference = hp['function_values'][:, 0] - hp['function_values'][:, 1]
        self.assertLess(difference.square().mean().item(), 1e-6)

    def test_clean_tasks_really_have_lower_intrinsic_label_entropy(self):
        _, _, _, hp = sample_classification_datasets(1024, 16, 2, generator=self.generator(3), return_diagnostics=True)
        p = hp['probabilities'].clamp(1e-12, 1-1e-12)
        entropy = -(p*p.log2() + (1-p)*(1-p).log2()).mean(1)
        self.assertLess(entropy[hp['clean']].mean().item(), .05)
        self.assertGreater(entropy[~hp['clean']].mean().item(), .1)

    def test_appended_bag_member_preserves_legacy_assignments(self):
        from small_mdlm.vendor.pfn_prior_bag import CODES, assign, parse_weights
        rng = np.random.default_rng(12)
        expected_rng = np.random.default_rng(12)
        expected = np.where(expected_rng.random(1000) < .5, 'tabpfn', 'tabpfn_gp')
        np.testing.assert_array_equal(assign(rng, 1000, parse_weights('tabpfn=0,tabpfn_gp=0')), expected)
        self.assertEqual(CODES['tabpfn_gp'], 5)
        self.assertEqual(CODES['controlled_gp'], 6)

    def test_new_member_fills_mixed_bank_and_leaves_one_d_feature_zero(self):
        from generate_pfn_bank import make_chunk
        from small_mdlm.vendor.pfn_prior_bag import parse_weights
        settings = {'lengthscale_min': 1.0, 'lengthscale_max': 2.0}
        job = (4, 34, .3, 0, None, None, parse_weights('tabpfn=0,controlled_gp=0'), 'cpu', 64, settings)
        result = make_chunk(job)
        _, x, latent, y, dims, codes = result
        self.assertEqual(set(np.unique(codes)), {3, 6})
        self.assertTrue(np.all(x[dims == 1, :, 1] == 0))
        self.assertTrue(np.isfinite(x).all() and np.isfinite(latent).all())
        self.assertTrue(set(np.unique(y)) <= {0, 1})
        again = make_chunk(job)
        for a, b in zip(result[1:], again[1:]):
            np.testing.assert_array_equal(a, b)

    def test_new_training_profile_matches_old_except_bank(self):
        root = Path(__file__).parent/'configs'
        old = json.loads((root/'tabpfn_gpmlp50_classify2d.json').read_text())
        new = json.loads((root/'tabpfn_controlled_gp50_classify2d.json').read_text())
        self.assertNotEqual(old['data']['bank_path'], new['data']['bank_path'])
        new['data']['bank_path'] = old['data']['bank_path']
        self.assertEqual(old, new)

    def test_legacy_chunk_ignores_new_settings(self):
        from generate_pfn_bank import make_chunk
        from small_mdlm.vendor.pfn_prior_bag import parse_weights
        old = (2, 34, .3, 0, None, None, parse_weights('tabpfn_gp=0'), 'cpu', 32)
        first = make_chunk(old)
        second = make_chunk((*old, {'clean_probability': 1.0, 'lengthscale_min': 1.0}))
        for a, b in zip(first[1:], second[1:]):
            np.testing.assert_array_equal(a, b)

    def test_legacy_conditional_evaluation_has_disjoint_splits(self):
        import tempfile
        from evaluate_tabpfn_conditionals import tasks_from_bank
        from test_pfn_bank import write_bank
        with tempfile.TemporaryDirectory() as folder:
            bank = Path(folder)
            write_bank(bank, total=100)
            (bank/'meta.json').write_text('{}')
            validation = tasks_from_bank(bank, 40, 'validation', 8, 2026)
            test = tasks_from_bank(bank, 40, 'test', 8, 2026)
            self.assertTrue(set(validation[5]).isdisjoint(test[5]))
            again = tasks_from_bank(bank, 40, 'test', 8, 2026)
            np.testing.assert_array_equal(test[5], again[5])
            for (cx, cy), xq, z, y, u in zip(*test[:5]):
                self.assertEqual(cx.shape, (len(cy), 2))
                self.assertEqual(xq.shape, (2,))
                self.assertEqual(z.shape, (2,))
                self.assertIn(y, (0, 1))
                self.assertIn(u, (0, 1))

    def test_posterior_reference_uses_shared_particles_for_conditionals(self):
        from screen_controlled_gp_prior import posterior_summary
        logs = torch.zeros(4, dtype=torch.float64)
        query = torch.tensor([[.9],[.1],[.9],[.1]], dtype=torch.float64)
        auxiliary = torch.tensor([[[.8]],[[.2]],[[.8]],[[.2]]], dtype=torch.float64)
        result = posterior_summary(logs, query, auxiliary)
        self.assertAlmostEqual(result['p_y'][0], .5)
        self.assertAlmostEqual(result['p_y_given_u1'][0][0], .74)
        self.assertAlmostEqual(result['p_y_given_u0'][0][0], .26)
        self.assertAlmostEqual(result['effective_sample_size'], 4)
        self.assertLess(result['marginal_consistency_max_abs_error'], 1e-12)
        self.assertFalse(result['conclusive'])


if __name__ == '__main__':
    unittest.main()
