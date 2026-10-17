"""Adapter contract tests without loading any checkpoint or using a GPU."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import small_mdlm_backend as adapter
from src.utils import extract


class VUDAdapterTests(unittest.TestCase):
    def test_response_is_prompt_continuation(self):
        backend = Mock()
        backend.generate.return_value = [1.25]
        prompt = 'x1 = -1 <output> 4 </output>\nx1 = 2 <output>'
        with patch.object(adapter, 'load', return_value=backend):
            output = adapter.generate_text(prompt, 7, 1.0, 'mdlm:test.pt')
        self.assertEqual(extract(output), 1.25)
        x, y, targets, seed, temperature = backend.generate.call_args.args
        self.assertEqual(x, [[-1.0], [2.0]])
        self.assertEqual(y[0], 4.0)
        self.assertTrue(np.isnan(y[1]))
        self.assertEqual((targets, seed, temperature), ([1], 7, 1.0))

    def test_joint_placeholder_and_target_order(self):
        backend = Mock()
        backend.generate.return_value = [2.0, 3.0]
        backend.tokenizer.number.return_value = [9, 12]
        icl = ('x1 = 0 <output> ###VUD_U_PLACEHOLDER### </output>\n'
               'x1 = 1 <output> 2 </output>')
        with patch.object(adapter, 'load', return_value=backend):
            result = adapter.sample_joint_numbers(icl, 'x1 = 3', 9, 1.0, 'mdlm:test.pt')
        self.assertEqual(result, (2.0, 3.0, [9, 12]))
        x, y, targets, seed, temperature = backend.generate.call_args.args
        self.assertEqual(x, [[0.0], [1.0], [3.0]])
        self.assertTrue(np.isnan(y[0]) and np.isnan(y[2]))
        self.assertEqual((targets, seed, temperature), ([0, 2], 9, 1.0))

    def test_classification_scores_query_labels(self):
        backend = Mock()
        backend.config = {'data': {'input_size': 1}}
        backend.label_probs.return_value = np.array([0.3, 0.7])
        prompt = ' x1 = -1.5 <output>0</output>\n x1 = 2.0 <output>1</output>\n x1 = 0.4 <output>'
        with patch.object(adapter, 'load', return_value=backend):
            label, probs = adapter.score_labels(prompt, ['0', '1'], 1.0, 'mdlm:test.pt')
        self.assertEqual((label, probs), ('1', {'0': 0.3, '1': 0.7}))
        x, y, target, temperature = backend.label_probs.call_args.args
        self.assertEqual(x, [[-1.5], [2.0], [0.4]])
        self.assertEqual(y[:2], [0.0, 1.0])
        self.assertTrue(np.isnan(y[2]))
        self.assertEqual((target, temperature), (2, 1.0))

    def test_classification_joint_labels_batch_per_context_targets(self):
        backend = Mock()
        backend.config = {'data': {'input_size': 2}}
        backend.sample_labels.return_value = [[1, 0], [0, 0]]
        icls = [' x1 = 0.5 <output>###VUD_U_PLACEHOLDER###</output>\n x1 = -1.0 <output>1</output>',
                ' x1 = -1.0 <output>1</output>\n x1 = 0.5 <output>###VUD_U_PLACEHOLDER###</output>']
        with patch.object(adapter, 'load', return_value=backend):
            result = adapter.sample_joint_labels(icls, 'x1 = 3.0', ['0', '1'], 9, 1.0, 'mdlm:test.pt')
        self.assertEqual(result, [('1', '0'), ('0', '0')])
        xs, ys, targets, seed, temperature = backend.sample_labels.call_args.args
        self.assertEqual(xs[0], [[0.5, 0.0], [-1.0, 0.0], [3.0, 0.0]])
        self.assertEqual(xs[1], [[-1.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
        self.assertTrue(np.isnan(ys[0][0]) and np.isnan(ys[1][1]) and np.isnan(ys[1][2]))
        self.assertEqual((targets, seed, temperature), ([[0, 2], [1, 2]], 9, 1.0))

    def test_classification_exact_joint_law_per_context_targets(self):
        backend = Mock()
        backend.config = {'data': {'input_size': 2}}
        joint = np.array([[0.54, 0.06], [0.10, 0.30]])
        backend.pair_label_law.return_value = [{'joint': joint, 'a_first': joint, 'b_first': joint.T}]
        icls = [' x1 = -1.0 <output>1</output>\n x1 = 0.5 <output>###VUD_U_PLACEHOLDER###</output>']
        with patch.object(adapter, 'load', return_value=backend):
            (law,) = adapter.joint_label_law(icls, 'x1 = 3.0', ['0', '1'], 1.0, 'mdlm:test.pt')
        self.assertEqual(law['joint'][('0', '1')], 0.06)
        self.assertEqual(law['y_first'][('0', '1')], 0.10)
        xs, ys, targets, temperature = backend.pair_label_law.call_args.args
        self.assertEqual(xs[0], [[-1.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
        self.assertEqual((targets, temperature), ([[1, 2]], 1.0))

    def test_exact_joint_statistics_match_count_estimator(self):
        from src.parallel.parallel_stats import joint_count_statistics, joint_distribution_statistics
        counts = {'0': {'0': 54, '1': 6}, '1': {'0': 10, '1': 30}}
        exact = joint_distribution_statistics({u: {y: c / 100 for y, c in row.items()} for u, row in counts.items()})
        self.assertAlmostEqual(exact['Va'], joint_count_statistics(counts)['Va'])
        self.assertAlmostEqual(exact['Va'], 0.606, places=3)
        self.assertAlmostEqual(exact['py']['0'], 0.64)

    def test_backward_estimator_reuses_prompts_and_inverts_by_bayes(self):
        from src.parallel.va_estimators import BackwardVaEstimator
        from src.utils import calculate_entropy
        prompts = {('z', None, None): {'0': 0.6, '1': 0.4},          # p(u|z,D)
                   ('z', 'x', '0'): {'0': 0.8, '1': 0.2},            # p(u|z,x,y=0,D)
                   ('z', 'x', '1'): {'0': 0.3, '1': 0.7}}            # p(u|z,x,y=1,D)
        experiment = Mock(label_keys=['0', '1'])
        experiment.calculate_avg_probs.side_effect = (
            lambda query_note, probability_calculated, icl_z_note=None, icl_u_label=None:
            prompts[(query_note, icl_z_note, icl_u_label)])
        pyx = {'0': 0.7, '1': 0.3}
        out = BackwardVaEstimator().estimate(experiment, 'x', 'z', pyx, calculate_entropy(pyx), 0.21)
        self.assertAlmostEqual(out['pB(y=1,u=0|x,z,D)'], 0.09)
        self.assertAlmostEqual(out['p(u=0|x,z,D)'], 0.65)
        self.assertAlmostEqual(out['p(y=0|x,u=1,z,D)'], 0.4)
        self.assertAlmostEqual(out['Va'], 0.717, places=3)
        self.assertGreaterEqual(out['Ve'], 0)
        self.assertAlmostEqual(out['p(y=0|x,z,D)'], 0.7)             # y-marginal is preserved
        self.assertGreater(out['kl_pyx_pyxz'], 0)                      # u-marginal check: 0.6 vs 0.65

    def test_gp_backend_zscores_context_and_matches_sklearn(self):
        import json, tempfile
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF
        from src import gp_backend
        with tempfile.TemporaryDirectory() as tmp:
            fit = f'{tmp}/fit.json'
            json.dump({'amplitude': 2.0, 'lengthscale': 0.7}, open(fit, 'w'))
            prompt = ' x1 = -2.0 <output>0</output>\n x1 = 1.0 <output>1</output>\n x1 = 4.0 <output>1</output>\n x1 = 0.5 <output>'
            label, probs = gp_backend.score_labels(prompt, ['0', '1'], 1.0, f'gp:{fit}')
        x = np.array([[-2.0], [1.0], [4.0]]); z = (x - x.mean()) / x.std()
        clf = GaussianProcessClassifier(ConstantKernel(2.0, 'fixed') * RBF(0.7, 'fixed'), optimizer=None).fit(z, [0, 1, 1])
        expected = clf.predict_proba((np.array([[0.5]]) - x.mean()) / x.std())[0]
        self.assertAlmostEqual(probs['1'], expected[1])
        self.assertAlmostEqual(probs['0'] + probs['1'], 1.0)
        self.assertEqual(label, max(probs, key=probs.get))

    def test_classification_pads_one_d_prompts_for_two_d_checkpoints(self):
        backend = Mock()
        backend.config = {'data': {'input_size': 2}}
        backend.label_probs.return_value = np.array([0.6, 0.4])
        prompt = ' x1 = 1.5 <output>1</output>\n x1 = -2.0 <output>'
        with patch.object(adapter, 'load', return_value=backend):
            adapter.score_labels(prompt, ['0', '1'], 1.0, 'mdlm:test.pt')
        self.assertEqual(backend.label_probs.call_args.args[0], [[1.5, 0.0], [-2.0, 0.0]])
        moons = ' x1 = 0.97; x2 = 0.51 <output>0</output>\n x1 = 1.2; x2 = -0.3 <output>'
        with patch.object(adapter, 'load', return_value=backend):
            adapter.score_labels(moons, ['0', '1'], 1.0, 'mdlm:test.pt')
        self.assertEqual(backend.label_probs.call_args.args[0], [[0.97, 0.51], [1.2, -0.3]])


if __name__ == '__main__':
    unittest.main()
