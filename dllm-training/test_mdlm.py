import unittest
import numpy as np
import torch
from small_mdlm.episodes import EpisodeConfig, episode
from small_mdlm.tokenizer import NumberTokenizer
from small_mdlm.model import build_model, masked_loss, sample


class MDLMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.tok = NumberTokenizer()
        self.model = build_model({'hidden_size': 32, 'cond_dim': 32,
                                 'n_blocks': 1, 'n_heads': 2, 'dropout': 0.0}, self.tok)

    def test_same_function_multiple_queries_and_reproducible_noise(self):
        cfg = EpisodeConfig(noise_std=0)
        for index in range(10):
            e = episode(cfg, 0, index)
            np.testing.assert_allclose(e['y'], e['x'] @ e['weights'] + e['bias'])
            np.testing.assert_array_equal(e['x'], episode(cfg, 0, index)['x'])
        cfg.noise_std = 2
        a, b = episode(cfg, 0, 0), episode(cfg, 0, 0)
        np.testing.assert_array_equal(a['y'], b['y'])
        self.assertFalse(np.array_equal(a['y'], a['x'] @ a['weights'] + a['bias']))

    def test_tokenizer_boundaries_and_hidden_targets(self):
        for value in (-999.99, -1.23, 0, 999.99):
            self.assertAlmostEqual(value, self.tok.decode_number(self.tok.number(value)))
        with self.assertRaises(ValueError):
            self.tok.number(1000)
        with self.assertRaises(ValueError):
            self.tok.number(float('inf'))
        a = self.tok.encode([[1], [2]], [3, float('nan')], [1], hide=True)
        b = self.tok.encode([[1], [2]], [3, 999], [1], hide=True)
        torch.testing.assert_close(a['ids'], b['ids'])
        self.assertEqual(int((a['ids'] == self.tok.mask_id).sum()), 6)

    def test_backward_sampling_and_conditioning(self):
        a = self.tok.encode([[1], [2]], [3, 6], [1])
        b = self.tok.encode([[1], [2], [3]], [3, 6, 9], [0, 2])
        ids, kinds, attention, _ = self.tok.collate([a, b])
        gen = torch.Generator().manual_seed(42)
        loss = masked_loss(self.model, self.tok, ids, kinds, attention, gen)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        generated = sample(self.model, self.tok, ids, kinds, attention, seed=1, steps=4)
        repeated = sample(self.model, self.tok, ids, kinds, attention, seed=1, steps=4)
        torch.testing.assert_close(generated, repeated)
        torch.testing.assert_close(generated[kinds == 0], ids[kinds == 0])
        self.assertFalse((generated == self.tok.mask_id).any())
        for i, e in enumerate((a, b)):
            for span in e['spans'].values():
                self.assertTrue(np.isfinite(self.tok.decode_number(generated[i, span])))

    def test_padding_is_ignored_by_attention(self):
        # Output layer starts at zero upstream; randomize it to exercise padding.
        with torch.no_grad():
            self.model.output_layer.linear.weight.normal_()
            for block in self.model.blocks:
                block.adaLN_modulation.bias.fill_(0.1)
        self.model.eval()
        a = self.tok.encode([[1], [2]], [3, 6], [1], hide=True)
        b = self.tok.encode([[1], [2], [3]], [3, 6, 9], [2], hide=True)
        ids, _, mask, _ = self.tok.collate([a])
        batch, _, batch_mask, _ = self.tok.collate([a, b])
        with torch.no_grad():
            alone = self.model(ids, torch.zeros(1), mask)
            padded = self.model(batch, torch.zeros(2), batch_mask)
        torch.testing.assert_close(alone[0], padded[0, :len(a['ids'])], atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
