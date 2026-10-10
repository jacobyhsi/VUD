import unittest
import numpy as np
import torch
from small_mdlm.bayes import logistic_posterior_predictive
from small_mdlm.episodes import EpisodeConfig, episode
from small_mdlm.model import answer_probs, build_model, masked_loss
from small_mdlm.tokenizer import NumberTokenizer

MODEL = {'hidden_size': 32, 'cond_dim': 32, 'n_blocks': 1, 'n_heads': 2, 'dropout': 0.0,
         'value_embedding': 'row', 'value_scale': 10.0}
CFG = EpisodeConfig(task='classification', cluster_prob=0.5, near_query_prob=0.7)


class ClassificationTests(unittest.TestCase):
    def test_label_tokens_rows_and_no_leak(self):
        tok = NumberTokenizer(labels=2)
        e = tok.encode([[0.5], [-1.0], [2.0]], [1, 0, 1], [2])
        (position,) = e['spans'][2]
        self.assertEqual(int(e['kinds'][position]), 3)
        self.assertEqual(int((e['kinds'] > 0).sum()), 1)
        # Row vector (x, onehot(y), is_hidden).
        rows = e['values'][:, 1:]
        self.assertTrue(((rows == torch.tensor([0.5, 0, 1, 0])).all(-1)).any())
        self.assertTrue(((rows == torch.tensor([-1.0, 1, 0, 0])).all(-1)).any())
        torch.testing.assert_close(rows[position], torch.tensor([2.0, 0, 0, 1]))
        hidden = tok.encode([[0.5], [2.0]], [1, float('nan')], [1], hide=True)
        leaked = tok.encode([[0.5], [2.0]], [1, 0], [1], hide=True)
        torch.testing.assert_close(hidden['ids'], leaked['ids'])
        torch.testing.assert_close(hidden['values'], leaked['values'])
        with self.assertRaises(ValueError):
            tok.label(2)
        # Regression tokenizers keep their row size.
        self.assertEqual(NumberTokenizer().row_size(1), 3)

    def test_episodes_follow_logistic_labels(self):
        y, p = [], []
        for index in range(400):
            e = episode(CFG, 0, index)
            self.assertTrue(set(np.unique(e['y'])) <= {0.0, 1.0})
            np.testing.assert_allclose(e['p'], 1 / (1 + np.exp(-(e['x'][:, 0] * e['weights'][0] + e['bias']))))
            y += list(e['y']); p += list(e['p'])
        y, p = np.array(y), np.array(p)
        self.assertAlmostEqual(y[p > 0.9].mean(), p[p > 0.9].mean(), delta=0.03)
        self.assertAlmostEqual(y[p < 0.1].mean(), p[p < 0.1].mean(), delta=0.03)

    def test_oracle_recovers_truth_and_splits_entropy(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-10, 10, 2000)
        y = (rng.random(2000) < 1 / (1 + np.exp(-0.8 * (x - 2)))).astype(float)
        out = logistic_posterior_predictive(CFG, x, y, np.array([-5.0, 2.0, 6.0]))
        np.testing.assert_allclose(out['p1'], 1 / (1 + np.exp(-0.8 * (np.array([-5, 2, 6]) - 2))), atol=0.03)
        self.assertLess(out['epistemic_entropy'].max(), 0.01)
        few = logistic_posterior_predictive(CFG, [-1.0, 1.0], [0.0, 1.0], np.array([0.0, 12.0]))
        np.testing.assert_allclose(few['total_entropy'], few['aleatoric_entropy'] + few['epistemic_entropy'])
        self.assertTrue((few['epistemic_entropy'] > 0.05).all())

    def test_model_trains_and_probs_normalize_over_labels(self):
        tok = NumberTokenizer(labels=2)
        model = build_model(MODEL, tok, input_size=1)
        with torch.no_grad():  # the DiT output layer starts at zero; open it so gradients flow
            model.output_layer.linear.weight.normal_()
        batch = [tok.encode(e['x'], e['y'], e['target_rows']) for e in (episode(CFG, 0, i) for i in range(4))]
        ids, kinds, attention, values = tok.collate(batch)
        generator = torch.Generator().manual_seed(0)
        loss = sum(masked_loss(model, tok, ids, kinds, attention, generator, values) for _ in range(8))
        loss.backward()
        self.assertGreater(model.row_embed.grad.abs().sum().item(), 0)
        probs = answer_probs(model, tok, ids, kinds, attention, values)
        answer = probs[kinds == 3]
        torch.testing.assert_close(answer[:, 12:14].sum(-1), torch.ones(len(answer)))
        self.assertLess(answer[:, 14:].sum().item(), 1e-6)

    def test_mixed_family_episodes(self):
        cfg = EpisodeConfig(task='classification_mix', input_size=2, near_query_prob=0.7,
                            context_min=3, context_max=32)
        tok = NumberTokenizer(labels=2)
        families, one_d, scales, agree = set(), 0, [], []
        for index in range(300):
            e = episode(cfg, 0, index)
            families.add(e['family'])
            self.assertEqual(e['x'].shape[1], 2)
            np.testing.assert_allclose(e['probs'].sum(1), 1, atol=1e-9)
            self.assertTrue(set(np.unique(e['y'])) <= {0.0, 1.0})
            if e['active_dims'] == 1:
                one_d += 1
                self.assertTrue(np.all(e['x'][:, 1] == 0))
            scales.append(np.abs(e['x']).max())
            agree += list(e['probs'][np.arange(len(e['y'])), e['y'].astype(int)])
            tok.encode(e['x'], e['y'], e['target_rows'])  # stays inside the tokenizer range
        self.assertEqual(families, {'linear', 'rff', 'curves'})
        self.assertTrue(50 < one_d < 150)
        self.assertLess(np.percentile(scales, 10), 3)
        self.assertGreater(np.percentile(scales, 90), 8)
        # Labels are drawn from probs: mean probability of the drawn label > 1/2.
        self.assertGreater(np.mean(agree), 0.7)

    def test_moons_truth_explains_the_vud_data(self):
        from evaluate_vud_classification import load, true_p1
        for name, bound in (('moons_1', 0.1), ('moons_2', 0.45)):
            x, y = load(name)
            p = np.clip(np.where(y == 1, true_p1(name, x), 1 - true_p1(name, x)), 1e-12, 1)
            self.assertLess(np.mean(-np.log(p)), bound)
        x, y = load('logistic_regression')
        self.assertAlmostEqual(np.mean(true_p1('logistic_regression', x)), y.mean(), delta=0.05)


if __name__ == '__main__':
    unittest.main()
