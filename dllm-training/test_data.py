import unittest

import numpy as np

from data import LinearRegressionTasks, RegressionConfig


class RegressionDataTests(unittest.TestCase):
    def test_shapes_and_function(self):
        for dim, size in ((10, 10), (1, 1), (3, 15)):
            task = LinearRegressionTasks(RegressionConfig(dim, size)).sample(0)
            self.assertEqual(task.context_x.shape, (size, dim))
            self.assertEqual(task.context_y.shape, (size,))
            self.assertEqual(task.query_x.shape, (1, dim))
            self.assertEqual(task.query_y.shape, (1,))
            np.testing.assert_allclose(task.context_y, task.context_x @ task.weights)
            np.testing.assert_allclose(task.query_y, task.query_x @ task.weights)
            self.assertTrue(np.all(np.abs(task.context_x) <= 0.5))
            self.assertTrue(np.all(np.abs(task.query_x) <= 0.5))

    def test_reproducibility_and_streams(self):
        source = LinearRegressionTasks(seed=7)
        first = source.sample(20)
        source.sample(8)
        np.testing.assert_array_equal(first.constructed_tokens(), source.sample(20).constructed_tokens())
        np.testing.assert_array_equal(first.weights, LinearRegressionTasks(seed=7).sample(20).weights)
        for other in (source.sample(21), LinearRegressionTasks(seed=8).sample(20),
                      LinearRegressionTasks(seed=7, split='validation').sample(20),
                      LinearRegressionTasks(seed=7, split='test').sample(20)):
            self.assertFalse(np.array_equal(first.weights, other.weights))

    def test_hidden_query_and_no_aliasing(self):
        task = LinearRegressionTasks().sample(0)
        self.assertEqual(set(task.model_inputs()), {'context_x', 'context_y', 'query_x'})
        tokens = task.constructed_tokens()
        self.assertEqual(tokens.shape, (11, 11))
        self.assertEqual(tokens[-1, -1], 0)
        np.testing.assert_array_equal(tokens[-1, :-1], task.query_x[0])
        inputs = task.model_inputs()
        inputs['context_y'][:] = 999
        self.assertFalse(np.any(task.context_y == 999))

    def test_config_and_invalid_arguments(self):
        task = LinearRegressionTasks(RegressionConfig(input_range=4, weight_scale=0)).sample(0)
        self.assertTrue(np.all(np.abs(task.context_x) <= 2))
        np.testing.assert_array_equal(task.context_y, np.zeros(10))
        for kwargs in ({'input_size': 0}, {'context_size': -1}, {'input_range': 0},
                       {'input_range': float('inf')}, {'weight_scale': -1}):
            with self.assertRaises(ValueError):
                RegressionConfig(**kwargs)
        with self.assertRaises(ValueError):
            LinearRegressionTasks(split='unknown')
        with self.assertRaises(ValueError):
            LinearRegressionTasks(seed=-1)
        with self.assertRaises(ValueError):
            LinearRegressionTasks().sample(-1)


if __name__ == '__main__':
    unittest.main()
