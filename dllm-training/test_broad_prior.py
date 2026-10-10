import unittest
import numpy as np
import torch
from small_mdlm.bayes import posterior_predictive
from small_mdlm.episodes import EpisodeConfig, episode
from small_mdlm.tokenizer import NumberTokenizer
from train import TrainBatches

BROAD = dict(noise_std=0.1, noise_std_max=3.0, cluster_prob=1.0, near_query_prob=1.0)


def original_episode(config, seed, index, split='train'):
    """The pre-broad-prior generator, kept verbatim to pin old behaviour."""
    split_id = {'train': 0, 'validation': 1, 'test': 2}[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, index]))
    n = int(rng.integers(config.context_min, config.context_max + 1))
    q = int(rng.integers(1, config.max_queries + 1))
    w = rng.normal(0, config.weight_scale, config.input_size)
    bias = rng.normal(0, config.bias_scale)
    x = rng.uniform(-config.input_range / 2, config.input_range / 2, (n + q, config.input_size))
    y = x @ w + bias + rng.normal(0, config.noise_std, n + q)
    targets = [n + q - 1]
    if q == 2:
        targets.insert(0, int(rng.integers(0, n + 1)))
    return x, y, targets


class BroadPriorTests(unittest.TestCase):
    def test_default_config_reproduces_original_episodes(self):
        for cfg in (EpisodeConfig(), EpisodeConfig(input_size=3, noise_std=0)):
            for index in range(20):
                e = episode(cfg, 4, index)
                x, y, targets = original_episode(cfg, 4, index)
                np.testing.assert_array_equal(e['x'], x)
                np.testing.assert_array_equal(e['y'], y)
                self.assertEqual(e['target_rows'], targets)

    def test_broad_episodes_follow_their_options(self):
        cfg = EpisodeConfig(**BROAD)
        near = []
        for index in range(200):
            e = episode(cfg, 0, index)
            x, rows = e['x'][:, 0], e['target_rows']
            self.assertEqual(rows[-1], len(x) - 1)
            self.assertTrue(0.1 <= e['noise_std'] <= 3.0)
            self.assertTrue(np.all(np.abs(x[-1]) <= 15))
            others = np.delete(x[:-1], rows[0]) if len(rows) == 2 else x[:-1]
            if len(rows) == 2:
                near.append(abs(x[rows[0]] - x[-1]))
            else:
                near.append(np.min(np.abs(others - x[-1])))
        # With near_query_prob=1 the auxiliary is always close to the query.
        self.assertLess(np.median(near), 1.0)
        spreads = [episode(cfg, 0, i)['x'][:-1, 0].std() for i in range(200)]
        uniform = [episode(EpisodeConfig(noise_std_max=3.0, noise_std=0.1), 0, i)['x'][:-1, 0].std()
                   for i in range(200)]
        self.assertLess(np.median(spreads), np.median(uniform))

    def test_oracle_matches_gaussian_process_form_and_recovers_truth(self):
        cfg = EpisodeConfig(noise_std=2.0)
        rng = np.random.default_rng(0)
        x = rng.normal(1, 2, (15, 1))
        y = -x[:, 0] + 3 + rng.normal(0, 2, 15)
        xq = np.array([[-15.0], [1.0], [15.0]])
        got = posterior_predictive(cfg, x, y, xq)
        # Same posterior via the kernel (GP) form: k(a, b) = ws^2 a b + bs^2.
        k = lambda a, b: 4.0 * a @ b.T + 9.0
        gram = k(x, x) + 4.0 * np.eye(15)
        mean = k(xq, x) @ np.linalg.solve(gram, y)
        var = np.diag(k(xq, xq) - k(xq, x) @ np.linalg.solve(gram, k(x, xq))) + 4.0
        np.testing.assert_allclose(got['mean'], mean, rtol=1e-8)
        np.testing.assert_allclose(got['total_var'], var, rtol=1e-8)
        np.testing.assert_allclose(got['aleatoric_var'], 4.0)
        # Far queries carry more epistemic uncertainty than the cluster centre.
        self.assertGreater(got['epistemic_var'][0], 10 * got['epistemic_var'][1])
        broad = EpisodeConfig(**BROAD)
        x = rng.uniform(-15, 15, (400, 1))
        y = 2 * x[:, 0] - 1 + rng.normal(0, 1.5, 400)
        big = posterior_predictive(broad, x, y, np.array([[5.0]]))
        self.assertAlmostEqual(big['mean'][0], 9.0, delta=0.5)
        self.assertAlmostEqual(big['noise_std_posterior_mean'], 1.5, delta=0.15)
        self.assertLess(big['epistemic_var'][0], 0.1)

    def test_worker_batches_match_serial_batches(self):
        cfg = EpisodeConfig(context_min=3, context_max=6, **BROAD)
        tok = NumberTokenizer()
        dataset = TrainBatches(cfg, tok, seed=0, batch_size=4, steps=6)
        serial = [dataset[s] for s in range(2, 6)]
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, sampler=range(2, 6),
                                             num_workers=2, prefetch_factor=2)
        for expected, got in zip(serial, loader):
            for a, b in zip(expected, got):
                torch.testing.assert_close(a, b)

    def test_general_regression_families(self):
        cfg = EpisodeConfig(task='regression_mix', context_max=32, **BROAD)
        tok = NumberTokenizer()
        z, families = {'linear': [], 'modes': [], 'smooth': []}, set()
        for index in range(600):
            e = episode(cfg, 0, index)
            families.add(e['family'])
            self.assertEqual(e['target_rows'][-1], len(e['x']) - 1)
            self.assertTrue(np.all(e['std'] > 0))
            z[e['family']] += list((e['y'] - e['mean']) / e['std'])
            tok.encode(e['x'], e['y'], e['target_rows'])  # stays inside the tokenizer range
        self.assertEqual(families, {'linear', 'modes', 'smooth'})
        for family, values in z.items():
            # mean/std are the true conditional moments, so standardized residuals have unit variance.
            self.assertAlmostEqual(np.var(values), 1.0, delta=0.15, msg=family)
            self.assertAlmostEqual(np.mean(values), 0.0, delta=0.1, msg=family)

    def test_gp_regression_is_one_function_plus_gaussian_noise(self):
        cfg = EpisodeConfig(task='regression_gp', noise_std=0.05, noise_std_max=3.0, cluster_prob=0.5,
                            multi_cluster_prob=0.3, near_query_prob=0.7, context_max=32)
        tok = NumberTokenizer()
        z, kinds, gaps, low_noise = [], set(), 0, 0
        for index in range(800):
            e = episode(cfg, 0, index)
            kinds.add(e['family'])
            z += list((e['y'] - e['mean']) / e['std'])
            low_noise += e['std'].min() < 0.3
            context = np.sort(np.delete(e['x'][:, 0], e['target_rows']))
            gaps += np.diff(context).max() > 5
            tok.encode(e['x'], e['y'], e['target_rows'])  # stays inside the tokenizer range
        z = np.array(z)
        self.assertEqual(kinds, {'none', 'rbf', 'periodic', 'rough'})
        # Standardized residuals are N(0, 1): one mean per x (a two-function
        # mixture would give excess kurtosis), with the stated noise std.
        self.assertAlmostEqual(z.mean(), 0, delta=0.05)
        self.assertAlmostEqual(z.var(), 1, delta=0.05)
        self.assertAlmostEqual(np.mean(z ** 4) / z.var() ** 2, 3, delta=0.3)
        self.assertGreater(gaps, 100)
        self.assertGreater(low_noise, 200)

    def test_new_layout_option_keeps_old_broad_episodes(self):
        cfg = EpisodeConfig(**BROAD)
        for index in range(20):
            np.testing.assert_array_equal(episode(cfg, 1, index)['x'],
                                          episode(EpisodeConfig(**BROAD, multi_cluster_prob=0.0), 1, index)['x'])


if __name__ == '__main__':
    unittest.main()
