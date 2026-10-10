import tempfile
import unittest
from pathlib import Path

import numpy as np

from small_mdlm.episodes import EpisodeConfig, episode, normalize_by_context, _load_bank
from small_mdlm.tokenizer import NumberTokenizer


def write_bank(root, total=200, seq_len=34):
    rng = np.random.default_rng(0)
    x = rng.random((total, seq_len, 2)).astype(np.float32)
    x[: total // 3, :, 1] = 0  # 1-D datasets
    latent = rng.normal(size=(total, seq_len)).astype(np.float32)
    np.save(root/'x.npy', x)
    np.save(root/'latent.npy', latent)
    np.save(root/'y.npy', (rng.random((total, seq_len)) < 1 / (1 + np.exp(-latent))).astype(np.int8))
    np.save(root/'dims.npy', np.where(np.arange(total) < total // 3, 1, 2).astype(np.int8))
    return x


class PFNBankTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.x = write_bank(Path(self.dir.name))
        self.cfg = EpisodeConfig(task='classification_pfn', input_size=2, bank_path=self.dir.name,
                                 bank_holdout=40, context_min=3, context_max=30, max_queries=2)

    def tearDown(self):
        # Release mmap handles before removing the temporary bank (NFS keeps
        # open unlinked arrays as .nfs files and otherwise refuses rmdir).
        _load_bank.cache_clear()
        self.dir.cleanup()

    def test_normalization_uses_only_context_rows(self):
        x = np.array([[0.1, 0.0], [0.5, 0.0], [0.9, 0.0], [5.0, 0.0]])
        z = normalize_by_context(x, [0, 1, 2])
        np.testing.assert_allclose(z[:3, 0].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(z[:3, 0].std(), 1)
        self.assertTrue(np.all(z[:, 1] == 0))  # constant feature stays 0
        moved = x.copy(); moved[3, 0] = -7.0   # the hidden row cannot change the others
        np.testing.assert_allclose(normalize_by_context(moved, [0, 1, 2])[:3], z[:3])

    def test_episodes_draw_subsets_and_respect_splits(self):
        tok = NumberTokenizer(labels=2)
        splits = {'train': set(), 'validation': set(), 'test': set()}
        for split in splits:
            for index in range(60):
                e = episode(self.cfg, 0, index, split)
                splits[split].add(e['bank_index'])
                rows, x = e['target_rows'], e['x']
                self.assertEqual(rows[-1], len(x) - 1)
                observed = [r for r in range(len(x)) if r not in rows]
                np.testing.assert_allclose(x[observed, 0].mean(), 0, atol=1e-9)
                self.assertTrue(set(np.unique(e['y'])) <= {0.0, 1.0})
                np.testing.assert_allclose(e['probs'].sum(1), 1)
                if e['bank_index'] < 200 // 3:
                    self.assertTrue(np.all(x[:, 1] == 0))
                tok.encode(x, e['y'], rows)
        self.assertTrue(max(splits['train']) < 160 <= min(splits['validation']))
        self.assertTrue(max(splits['validation']) < 180 <= min(splits['test']))

    def test_local_context_puts_queries_beyond_the_context(self):
        local = EpisodeConfig(task='classification_pfn', input_size=2, bank_path=self.dir.name, bank_holdout=40,
                              context_min=8, context_max=12, max_queries=1, pfn_local_context_prob=1.0)
        spread = []
        for index in range(100):
            e = episode(local, 0, index)
            x, rows = e['x'], e['target_rows']
            # Context is z-scored, so a query outside the local region lands far out.
            spread.append(np.abs(x[rows[-1]]).max())
        self.assertGreater(np.median(spread), 2.0)
        # Off by default: the old random split, drawing no extra random numbers.
        np.testing.assert_array_equal(episode(self.cfg, 3, 5)['x'],
                                      episode(EpisodeConfig(**{**self.cfg.__dict__, 'pfn_local_context_prob': 0.0}), 3, 5)['x'])

    def test_prior_bag_members_and_weights(self):
        from small_mdlm.vendor.pfn_prior_bag import apply_labels, assign, member_probabilities, parse_weights
        weights = parse_weights('median=0,bernoulli=0')
        self.assertEqual(member_probabilities(weights)[1].tolist(), [0.5, 0.5])
        names, p = member_probabilities(parse_weights('bernoulli=0,median=1'))
        self.assertEqual(names, ['median', 'bernoulli'])
        np.testing.assert_allclose(p, np.exp([1, 0]) / np.exp([1, 0]).sum())
        with self.assertRaises(ValueError):
            parse_weights('threshold=1')
        rng = np.random.default_rng(0)
        members = assign(rng, 4000, weights)
        self.assertAlmostEqual((members == 'median').mean(), 0.5, delta=0.03)
        self.assertTrue(np.all(assign(np.random.default_rng(1), 50, parse_weights('bernoulli=0')) == 'bernoulli'))
        latent = rng.normal(size=(6, 34)).astype(np.float32)
        y = (rng.random((6, 34)) < 0.5).astype(np.int8)
        members = np.array(['median', 'bernoulli'] * 3)
        before = y.copy()
        median = apply_labels(members, latent, y)
        np.testing.assert_array_equal(y[~median], before[~median])  # Bernoulli labels untouched
        self.assertTrue(np.all(y[median].mean(1) == 0.5))           # median threshold balances 34 points
        self.assertTrue(np.all((latent[median] > 0) == (y[median] == 1)))

    def test_mlp_members_fill_a_chunk(self):
        from generate_pfn_bank import CHUNK, make_chunk
        from small_mdlm.vendor.pfn_prior_bag import parse_weights
        bag = parse_weights('bnn=0,tabpfn=0')
        index, x, latent, y, dims, codes = make_chunk((3, 34, 0.3, 0, None, None, bag, 'cpu', CHUNK))
        self.assertEqual(x.shape, (CHUNK, 34, 2))
        self.assertTrue(set(np.unique(codes)) == {2, 3})
        self.assertTrue(np.all(x[dims == 1, :, 1] == 0))           # 1-D datasets leave x2 at 0
        self.assertTrue(np.isfinite(x).all() and np.isfinite(latent).all())
        self.assertTrue(set(np.unique(y)) <= {0, 1})
        # Labels agree with the true probabilities most of the time.
        p = 1 / (1 + np.exp(-latent))
        self.assertGreater(np.mean((p > 0.5) == (y == 1)), 0.7)
        again = make_chunk((3, 34, 0.3, 0, None, None, bag, 'cpu', CHUNK))
        np.testing.assert_array_equal(x, again[1])                  # reproducible per chunk index

    def test_tabpfn_causal_member(self):
        from generate_pfn_bank import make_chunk
        from small_mdlm.vendor.pfn_prior_bag import parse_weights
        _, x, latent, y, dims, codes = make_chunk((1, 34, 0.3, 0, None, None, parse_weights('tabpfn_causal=0'), 'cpu', 64))
        self.assertTrue(np.all(codes == 4))
        self.assertTrue(np.isfinite(x).all() and np.abs(x).max() <= 100)  # outliers removed, then normalized
        self.assertTrue(np.all(x[dims == 1, :, 1] == 0))
        np.testing.assert_array_equal(latent > 0, y == 1)          # latents are the saturated labels

    def test_tabpfn_gp_member(self):
        from generate_pfn_bank import make_chunk
        from small_mdlm.vendor.pfn_prior_bag import parse_weights
        _, x, latent, y, dims, codes = make_chunk((2, 34, 0.3, 0, None, None, parse_weights('tabpfn_gp=0'), 'cpu', 64))
        self.assertTrue(np.all(codes == 5) and np.isfinite(x).all() and np.isfinite(latent).all())
        p = 1 / (1 + np.exp(-latent.astype(float)))
        self.assertGreater(np.mean((p > 0.5) == (y == 1)), 0.9)    # mostly clean threshold labels


if __name__ == '__main__':
    unittest.main()
