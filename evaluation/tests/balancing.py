"""
Balancing Test for Synthetic vs. Real Data

This script evaluates the balance of categorical and continuous features between a real dataset and multiple synthetic datasets. 
It calculates key metrics such as Total Variation Distance (TVD) and Jensen-Shannon Divergence (JSD) for categorical columns, 
and Skewness Difference (difference between real and fake skewness) and Wasserstein Distance for continuous columns.

Good balance is indicated by low divergence values (low TVD, JSD, and Wasserstein Distance) and a small Skewness Difference. 
High values suggest a poor match between the synthetic and real data distributions.

Usage:
1. **Direct Execution**:
   - Run the script from the command line with the required arguments, for example:
     ```
     python balancing.py real_data.csv dataset_name model_name output_dir "cat1,cat2" "con1,con2"
     ```
   - This will generate a JSON file with metrics for each synthetic dataset and an overall average.

2. **Importing in Another Script**:
   - Import the `balancing_main` function:
     ```
     from balancing import balancing_main
     ```
   - Call the function with appropriate arguments:
     ```
     balancing_main(real_path, fake_paths, dataset_name, model_name, performance_dir, categorical_columns, continuous_columns)
     ```
   - `real_path`: Path to the real dataset CSV file.
   - `fake_paths`: List of paths to synthetic dataset CSV files.
   - `dataset_name`: Name of the dataset.
   - `model_name`: Name of the synthetic data generation model.
   - `performance_dir`: Directory where the output JSON file will be saved.
   - `categorical_columns`: List of categorical columns (up to 5).
   - `continuous_columns`: List of continuous columns (up to 5).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, skew
import json
import os

def load_data(real_path, fake_path):
    real_data = pd.read_csv(real_path)
    fake_data = pd.read_csv(fake_path)
    
    # Ensure both datasets have the same columns
    common_columns = real_data.columns.intersection(fake_data.columns)
    real_data = real_data[common_columns]
    fake_data = fake_data[common_columns]

    real_data = real_data.head(10000)
    fake_data = fake_data.head(10000)
    
    return real_data, fake_data

def skewness_balance(real_data, fake_data, class_column):
    real_skew = skew(real_data[class_column].dropna())
    fake_skew = skew(fake_data[class_column].dropna())
    return {'Real Skewness': real_skew, 'Fake Skewness': fake_skew}

def jensen_shannon_divergence_balance(real_data, fake_data, class_column):
    p = real_data[class_column].value_counts(normalize=True, sort=False)
    q = fake_data[class_column].value_counts(normalize=True, sort=False)
    p, q = p.align(q, fill_value=0)
    js_div = jensenshannon(p, q)
    return {'Jensen-Shannon Divergence': js_div}

def wasserstein_distance_balance(real_data, fake_data, class_column):
    w_dist = wasserstein_distance(real_data[class_column].dropna(), fake_data[class_column].dropna())
    return {'Wasserstein Distance': w_dist}

def calculate_balance_metrics(real_data, fake_data, class_column):
    skewness_vals = skewness_balance(real_data, fake_data, class_column)
    js_divergence = jensen_shannon_divergence_balance(real_data, fake_data, class_column)
    wasserstein_dist = wasserstein_distance_balance(real_data, fake_data, class_column)

    # Create a dictionary of average metrics
    balance_metrics = {
        'Skewness': skewness_vals,
        'Jensen-Shannon Divergence': js_divergence,
        'Wasserstein Distance': wasserstein_dist,
    }
    
    return balance_metrics

def evaluate_and_save_results(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, class_column):
    all_metrics = []
    detailed_jsons = []

    for i, fake_path in enumerate(fake_paths, 1):
        real_data, fake_data = load_data(real_path, fake_path)
        
        # Calculate metrics for the current fake dataset
        metrics = calculate_balance_metrics(real_data, fake_data, class_column)
        
        # Append metrics for detailed results
        detailed_jsons.append({f"Fake dataset {i}": metrics})
        
        # Store the metrics for later processing
        all_metrics.append(metrics)

    # Save the combined detailed results into one JSON file
    detailed_output_filename = os.path.join(performance_dir, f"{tds_model_name}_{dataset_name}_balancing_evaluation_detailed.json")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_jsons, f, indent=4)

    # Calculate and save the overall averages
    overall_metrics = {}
    for key in ['Skewness', 'Jensen-Shannon Divergence', 'Wasserstein Distance']:
        real_values = [m[key]['Real Skewness'] if key == 'Skewness' else m[key] for m in all_metrics]
        fake_values = [m[key]['Fake Skewness'] if key == 'Skewness' else m[key] for m in all_metrics]
        overall_metrics[f"Average Real {key}"] = np.mean(real_values)
        overall_metrics[f"Average Fake {key}"] = np.mean(fake_values)
    
    overall_avg_df = pd.DataFrame([overall_metrics])
    overall_avg_output_filename = os.path.join(performance_dir, f"{tds_model_name}_{dataset_name}_balancing_overall_average.csv")
    overall_avg_df.to_csv(overall_avg_output_filename, index=False)

def balancing_main(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, class_column):
    evaluate_and_save_results(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, class_column)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate class balance in synthetic data")
    parser.add_argument("real_path", type=str, help="Path to the real data CSV file")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("tds_model_name", type=str, help="Name of the TDS model")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    parser.add_argument("class_column", type=str, help="Class column to evaluate")
    args = parser.parse_args()

    # Find all fake dataset paths
    fake_paths = [os.path.join(args.performance_dir, f"{args.tds_model_name}_{args.dataset_name}_{i}.csv") for i in range(1, 6)]

    balancing_main(args.real_path, fake_paths, args.dataset_name, args.tds_model_name, args.performance_dir, args.class_column)