import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from simulate_recover import sample_ez_parameters, compute_predicted_stats, recover_parameters, compute_bias, run_experiment

class EZDiffusionTestSuite(unittest.TestCase):
    def test_parameter_sampling(self):
        boundary, drift, nondecision = sample_ez_parameters()
        self.assertTrue(0.5 <= boundary <= 2)
        self.assertTrue(0.5 <= drift <= 2)
        self.assertTrue(0.1 <= nondecision <= 0.5)

    def test_recovery_consistency(self):
        true_values = (1.0, 1.0, 0.3)
        predicted = compute_predicted_stats(*true_values)
        estimated = recover_parameters(*predicted)
        bias, _ = compute_bias(true_values, estimated)

        self.assertAlmostEqual(bias[0], 0, places=2)
        self.assertAlmostEqual(bias[1], 0, places=2)
        self.assertAlmostEqual(bias[2], 0, places=2)

def test_stability_across_samples(self):
    sample_sizes = [10, 40, 4000]
    
    for size in sample_sizes:
        df = run_experiment(size, iterations=1000)
        mean_bias = df[["Bias Boundary", "Bias Drift", "Bias Nondecision"]].mean().values
        print(f"\nSample Size {size} - Mean Bias: {mean_bias}")

        if size == 10:
            boundary_tolerance = 0.12
            drift_tolerance = 2.8
            nondecision_tolerance = 0.35
        elif size == 40:
            boundary_tolerance = 0.1
            drift_tolerance = 1.0
            nondecision_tolerance = 0.3  
        else: 
            boundary_tolerance = 0.03
            drift_tolerance = 0.15
            nondecision_tolerance = 0.07

        self.assertAlmostEqual(mean_bias[0], 0, delta=boundary_tolerance)
        self.assertAlmostEqual(mean_bias[1], 0, delta=drift_tolerance)
        self.assertAlmostEqual(mean_bias[2], 0, delta=nondecision_tolerance)


if __name__ == "__main__":
    unittest.main()
