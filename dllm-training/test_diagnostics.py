import unittest
import numpy as np
import torch
from continuous_baseline import ContinuousICL, baselines, sample_tasks
from small_mdlm.model import all_masked_losses, build_model, masked_loss
from small_mdlm.tokenizer import NumberTokenizer

MODEL = {'hidden_size': 32, 'cond_dim': 32, 'n_blocks': 1, 'n_heads': 2, 'dropout': 0.0}
DATA = {'input_size': 3, 'context_size': 5, 'input_range': 1.0, 'weight_scale': 1.0, 'noise_std': 0.0}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_tasks_are_noiseless_linear_and_deterministic(self):
        x, y, xq, yq = sample_tasks(DATA, 0, 'train', 1, 4)
        self.assertEqual(x.shape, (4, 5, 3))
        self.assertTrue(np.abs(x).max() <= 0.5)
        w = np.stack([np.linalg.lstsq(x[i], y[i], rcond=None)[0] for i in range(4)])
        np.testing.assert_allclose(np.einsum('bd,bd->b', xq, w), yq, atol=1e-8)
        np.testing.assert_array_equal(x, sample_tasks(DATA, 0, 'train', 1, 4)[0])
        ref = baselines(x, y, xq, yq)
        self.assertLess(ref['least_squares_mse'], 1e-12)
        self.assertLessEqual(ref['gd1_mse'], ref['zero_mse'] + 1e-12)

    def test_continuous_model_trains_on_query_output(self):
        x, y, xq, yq = (torch.as_tensor(a, dtype=torch.float32) for a in sample_tasks(DATA, 0, 'train', 1, 4))
        for layout in ('constructed', 'classic'):
            model = ContinuousICL(MODEL, 3, layout)
            pred = model(x, y, xq)
            self.assertEqual(pred.shape, (4,))
            ((pred - yq) ** 2).mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_classic_layout_interleaves_x_and_y_tokens(self):
        model = ContinuousICL(MODEL, 2, 'classic')
        seen = {}
        model.dit.vocab_embed.register_forward_hook(lambda m, i, o: seen.update(tokens=i[0]))
        model(torch.tensor([[[1., 2.], [3., 4.]]]), torch.tensor([[5., 6.]]), torch.tensor([[7., 8.]]))
        expected = torch.tensor([[1, 2, 0, 0, 0], [0, 0, 5, 1, 0], [3, 4, 0, 0, 0],
                                 [0, 0, 6, 1, 0], [7, 8, 0, 0, 1]], dtype=torch.float32)
        torch.testing.assert_close(seen['tokens'][0], expected)

    def test_all_masked_losses_cover_answer_digits(self):
        tok = NumberTokenizer(integer_digits=1)
        model = build_model(MODEL, tok)
        e = tok.encode([[0.1], [0.2]], [0.3, -0.6], [1])
        ids, kinds, attention, _ = tok.collate([e])
        ce = all_masked_losses(model, tok, ids, kinds, attention)
        self.assertEqual(ce.shape, ids.shape)
        answer = ce[kinds > 0]
        self.assertEqual(len(answer), 4)  # sign, one integer digit, two decimals
        # Untrained constrained head: uniform over +/- and over ten digits.
        torch.testing.assert_close(answer, torch.tensor([np.log(2)] + [np.log(10)] * 3, dtype=torch.float32))

    def test_values_mark_observed_numbers_and_never_hidden_answers(self):
        tok = NumberTokenizer(integer_digits=1)
        e = tok.encode([[0.8], [-1.2], [0.5]], [-1.2, 1.8, -0.75], [2])
        width = tok.width
        scalar = e['values'][:, 0]
        # Every token of an observed number carries that number's value.
        for value in (0.8, -1.2, 1.8, 0.5):
            self.assertEqual(int((scalar == value).sum()), width * (2 if value == -1.2 else 1))
        self.assertTrue((scalar[e['kinds'] > 0] == 0).all())
        # Row vector (x, y, is_hidden): observed rows carry y, the hidden row
        # carries its x and the flag, and every answer token sees the query x.
        rows = e['values'][:, 1:]
        answer = rows[e['kinds'] > 0]
        torch.testing.assert_close(answer, torch.tensor([[0.5, 0.0, 1.0]]).expand_as(answer))
        self.assertEqual(int((rows == torch.tensor([-1.2, 1.8, 0.0])).all(-1).sum()), 6 + 2 * width)  # x = <x> ; y = <y> ;
        torch.testing.assert_close(rows[0], torch.zeros(3))
        torch.testing.assert_close(rows[-1], torch.zeros(3))
        hidden = tok.encode([[0.8], [0.5]], [-1.2, float('nan')], [1], hide=True)
        leaked = tok.encode([[0.8], [0.5]], [-1.2, 9.99], [1], hide=True)
        torch.testing.assert_close(hidden['values'], leaked['values'])

    def test_value_embedding_changes_logits_and_trains(self):
        tok = NumberTokenizer(integer_digits=1)
        plain = build_model(MODEL, tok)
        model = build_model(dict(MODEL, value_embedding=True, value_scale=1.0), tok)
        self.assertFalse(hasattr(plain, 'value_embed'))
        # DiT residual gates start at zero, so answers only see context once
        # the gates are nonzero; open them as the padding test does.
        with torch.no_grad():
            model.output_layer.linear.weight.normal_()
            for block in model.blocks:
                block.adaLN_modulation.bias.fill_(0.1)
        e = tok.encode([[0.8], [0.5]], [1.2, 0], [1])
        ids, kinds, attention, values = tok.collate([e, e])
        # Identical symbols; only the value channel differs (sign flipped).
        values[1] = -values[1]
        ce = all_masked_losses(model, tok, ids, kinds, attention, values)
        answer = kinds[0] > 0
        self.assertFalse(torch.allclose(ce[0, answer], ce[1, answer]))
        # Several masking draws so at least one masks an answer token.
        generator = torch.Generator().manual_seed(0)
        loss = sum(masked_loss(model, tok, ids, kinds, attention, generator, values) for _ in range(8))
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(model.value_embed.grad.abs().sum().item(), 0)

    def test_row_embedding_gives_answers_the_query_x(self):
        tok = NumberTokenizer(integer_digits=1)
        model = build_model(dict(MODEL, value_embedding='row', value_scale=1.0), tok, input_size=1)
        with torch.no_grad():
            model.output_layer.linear.weight.normal_()
        e = tok.encode([[0.8], [0.5]], [1.2, 0], [1])
        ids, kinds, attention, values = tok.collate([e, e])
        values[1, :, 1] = -values[1, :, 1]  # flip only the row's x channel
        # Blocks are identity at init, so answers can only differ through
        # their own row vector, which must include the query x.
        ce = all_masked_losses(model, tok, ids, kinds, attention, values)
        answer = kinds[0] > 0
        self.assertFalse(torch.allclose(ce[0, answer], ce[1, answer]))
        generator = torch.Generator().manual_seed(0)
        loss = sum(masked_loss(model, tok, ids, kinds, attention, generator, values) for _ in range(8))
        loss.backward()
        self.assertGreater(model.row_embed.grad.abs().sum().item(), 0)

    def test_cond_dim_defaults_to_zero(self):
        tok = NumberTokenizer()
        config = {k: v for k, v in MODEL.items() if k != 'cond_dim'}
        model = build_model(dict(config, value_embedding='row', value_scale=1.0), tok, input_size=1)
        # The timestep path is empty; adaLN reduces to its (zero-initialized) bias.
        self.assertEqual(sum(p.numel() for p in model.sigma_map.parameters()), 0)
        self.assertEqual(model.blocks[0].adaLN_modulation.weight.numel(), 0)
        self.assertEqual(model.blocks[0].adaLN_modulation.bias.abs().sum().item(), 0)
        with torch.no_grad():
            model.output_layer.linear.weight.normal_()
        ids, kinds, attention, values = tok.collate([tok.encode([[1.0], [2.0]], [3.0, 4.0], [1])])
        generator = torch.Generator().manual_seed(0)
        sum(masked_loss(model, tok, ids, kinds, attention, generator, values) for _ in range(8)).backward()
        self.assertGreater(model.blocks[0].adaLN_modulation.bias.grad.abs().sum().item(), 0)
        # Explicit cond_dim (older configs and checkpoints) is kept.
        self.assertEqual(build_model(MODEL, tok).sigma_map.mlp[0].out_features, MODEL['cond_dim'])


if __name__ == '__main__':
    unittest.main()
