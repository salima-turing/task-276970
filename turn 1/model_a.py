import unittest
import math

def compute_cross_entropy(predictions, labels):
   if len(predictions) != len(labels):
   	raise ValueError("Predictions and labels must have the same length.")

   cross_entropy = 0
   for pred, label in zip(predictions, labels):
   	if pred == 0:
   		pred = 1e-15  # Avoid log(0)
   	cross_entropy += -label * math.log(pred) - (1 - label) * math.log(1 - pred)

   return cross_entropy / len(predictions)

class TestCrossEntropy(unittest.TestCase):

   def test_compute_cross_entropy(self):
   	predictions = [0.9, 0.1, 0.8, 0.2]
   	labels = [1, 0, 1, 0]
   	expected_cross_entropy = 0.4689955149470922
   	computed_cross_entropy = compute_cross_entropy(predictions, labels)
   	self.assertAlmostEqual(computed_cross_entropy, expected_cross_entropy, places=6)

   def test_boundary_case(self):
   	predictions = [1.0]
   	labels = [1]
   	expected_cross_entropy = 0
   	computed_cross_entropy = compute_cross_entropy(predictions, labels)
   	self.assertEqual(computed_cross_entropy, expected_cross_entropy)

   def test_value_error(self):
   	predictions = [0.1, 0.2]
   	labels = [0, 1, 0]
   	with self.assertRaises(ValueError):
   		compute_cross_entropy(predictions, labels)

if __name__ == '__main__':
   unittest.main()
