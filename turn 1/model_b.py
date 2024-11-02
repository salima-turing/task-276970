import unittest
import numpy as np

def cross_entropy(y_true, y_pred):
    # Implementation of your cross-entropy function
    pass

class TestCrossEntropy(unittest.TestCase):

    def test_cross_entropy_known_output(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
        expected_output = 0.05646888888888889
        computed_output = cross_entropy(y_true, y_pred)
        self.assertAlmostEqual(computed_output, expected_output, places=6)

    def test_cross_entropy_edge_case_zero_probability(self):
        y_true = np.array([1, 0])
        y_pred = np.array([1.0, 0.0])
        expected_output = 0.0
        computed_output = cross_entropy(y_true, y_pred)
        self.assertEqual(computed_output, expected_output)

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
