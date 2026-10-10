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
