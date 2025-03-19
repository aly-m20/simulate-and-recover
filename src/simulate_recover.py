import numpy as np
import pandas as pd
import scipy.stats as stats

def sample_ez_parameters():
    boundary = np.random.uniform(0.5, 2)
    drift = np.random.uniform(0.5, 2)
    nondecision_time = np.random.uniform(0.1, 0.5)
    return boundary, drift, nondecision_time

def compute_predicted_stats(boundary, drift, nondecision_time):
    y = np.exp(-boundary * drift)
    predicted_accuracy = 1 / (y + 1)
    predicted_mean_rt = nondecision_time + (boundary / (2 * drift)) * ((1 - y) / (1 + y))
    predicted_rt_variance = (boundary / (2 * drift**3)) * ((1 - 2 * boundary * drift * y - y**2) / (y + 1)**2)
    return predicted_accuracy, predicted_mean_rt, predicted_rt_variance

def simulate_observations(true_accuracy, true_mean_rt, true_variance, sample_size):
    observed_correct_trials = np.random.binomial(sample_size, true_accuracy)
    observed_accuracy = observed_correct_trials / sample_size
    observed_mean_rt = np.random.normal(true_mean_rt, np.sqrt(true_variance / sample_size))
    observed_variance = stats.gamma.rvs((sample_size - 1) / 2, scale=(2 * true_variance / (sample_size - 1)))
    return observed_accuracy, observed_mean_rt, observed_variance

def recover_parameters(observed_accuracy, observed_mean_rt, observed_variance):
    epsilon = 1e-6  
    observed_accuracy = np.clip(observed_accuracy, epsilon, 1 - epsilon)  

    logit_acc = np.log(observed_accuracy / (1 - observed_accuracy))

    inside_term = max(logit_acc * (observed_accuracy**2 * logit_acc - observed_accuracy * logit_acc + observed_accuracy - 0.5) / (observed_variance + epsilon), epsilon)
    estimated_drift = np.sign(observed_accuracy - 0.5) * np.sqrt(inside_term)  

    if abs(estimated_drift) < epsilon or np.isnan(estimated_drift):
        estimated_drift = 1.0  

    estimated_boundary = logit_acc / (estimated_drift + epsilon)  
    estimated_nondecision = observed_mean_rt - (estimated_boundary / (2 * estimated_drift + epsilon)) * ((1 - np.exp(-estimated_drift * estimated_boundary)) / (1 + np.exp(-estimated_drift * estimated_boundary)))

    return estimated_boundary, estimated_drift, estimated_nondecision

def compute_bias(true_values, estimated_values):
    bias_values = np.array(true_values) - np.array(estimated_values)
    squared_error = bias_values ** 2
    return bias_values, squared_error

def run_experiment(sample_size, iterations=1000):
    results = []
    for _ in range(iterations):
        true_values = sample_ez_parameters()
        predicted_stats = compute_predicted_stats(*true_values)
        observed_stats = simulate_observations(*predicted_stats, sample_size)
        estimated_values = recover_parameters(*observed_stats)
        bias, error = compute_bias(true_values, estimated_values)
        results.append([sample_size] + list(true_values) + list(estimated_values) + list(bias) + list(error))
    
    columns = ["Sample Size", "True Boundary", "True Drift", "True Nondecision", 
               "Estimated Boundary", "Estimated Drift", "Estimated Nondecision",
               "Bias Boundary", "Bias Drift", "Bias Nondecision",
               "Error Boundary", "Error Drift", "Error Nondecision"]
    
    return pd.DataFrame(results, columns=columns)

if __name__ == "__main__":
    sizes = [10, 40, 4000]
    all_results = pd.concat([run_experiment(size) for size in sizes])
    all_results.to_csv("results/simulation_results.csv", index=False)