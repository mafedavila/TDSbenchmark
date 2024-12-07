"""
Augmentation Test for Synthetic vs. Real Data

This script evaluates how well a synthetic data generation tool performs in creating augmented datasets. 
It compares the real dataset with augmented synthetic datasets using statistical properties like Mean and Variance, 
as well as divergence metrics such as Jensen-Shannon Divergence (JSD) and Wasserstein Distance.

Good augmentation is indicated by similar statistical properties (Mean, Variance) between the real and augmented data, 
and low divergence values (low JSD and Wasserstein Distance), suggesting realistic and useful data augmentation.

Usage:
1. **Direct Execution**:
   - Run the script from the command line with the required arguments, for example:
     ```
     python augmentation.py real_data.csv dataset_name model_name output_dir "cat1,cat2" "con1,con2"
     ```
   - This will generate a JSON file with metrics for each augmented dataset and an overall average.

2. **Importing in Another Script**:
   - Import the `augmentation_main` function:
     ```
     from augmentation import augmentation_main
     ```
   - Call the function with appropriate arguments:
     ```
     augmentation_main(real_path, fake_paths, dataset_name, model_name, performance_dir, categorical_columns, continuous_columns)
     ```
   - `real_path`: Path to the real dataset CSV file.
   - `fake_paths`: List of paths to augmented dataset CSV files.
   - `dataset_name`: Name of the dataset.
   - `model_name`: Name of the synthetic data generation model.
   - `performance_dir`: Directory where the output JSON file will be saved.
   - `categorical_columns`: List of categorical columns (up to 5).
   - `continuous_columns`: List of continuous columns (up to 5).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
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

def select_columns(categorical_columns, continuous_columns):
    # Select up to 5 columns from each list
    selected_categorical = categorical_columns[:5]
    selected_continuous = continuous_columns[:5]

    # Check if there are any categorical or continuous columns
    if not selected_categorical:
        print("No categorical columns found.")
    if not selected_continuous:
        print("No continuous columns found.")

    return selected_categorical, selected_continuous

def mean_difference(real_data, fake_data, column):
    real_mean = real_data[column].mean()
    fake_mean = fake_data[column].mean()
    mean_diff = real_mean - fake_mean
    return {'Real Mean': real_mean, 'Fake Mean': fake_mean, 'Mean Difference': mean_diff}

def variance_difference(real_data, fake_data, column):
    real_variance = real_data[column].var()
    fake_variance = fake_data[column].var()
    variance_diff = real_variance - fake_variance
    return {'Real Variance': real_variance, 'Fake Variance': fake_variance, 'Variance Difference': variance_diff}

def jensen_shannon_divergence(real_data, fake_data, column):
    p = real_data[column].value_counts(normalize=True, sort=False)
    q = fake_data[column].value_counts(normalize=True, sort=False)
    p, q = p.align(q, fill_value=0)
    js_div = jensenshannon(p, q)
    return {'Jensen-Shannon Divergence': js_div}

def wasserstein_distance_metric(real_data, fake_data, column):
    w_dist = wasserstein_distance(real_data[column].dropna(), fake_data[column].dropna())
    return {'Wasserstein Distance': w_dist}

def calculate_augmentation_metrics(real_data, fake_data, categorical_columns, continuous_columns):
    metrics = {}

    if categorical_columns:
        for column in categorical_columns:
            if column in real_data.columns and column in fake_data.columns:
                js_divergence = jensen_shannon_divergence(real_data, fake_data, column)
                metrics[column] = {
                    'js_divergence': js_divergence['Jensen-Shannon Divergence']
                }
    
    if continuous_columns:
        for column in continuous_columns:
            if column in real_data.columns and column in fake_data.columns:
                mean_diff = mean_difference(real_data, fake_data, column)
                variance_diff = variance_difference(real_data, fake_data, column)
                wasserstein_dist = wasserstein_distance_metric(real_data, fake_data, column)
                metrics[column] = {
                    'mean_difference': mean_diff,
                    'variance_difference': variance_diff,
                    'wasserstein_dist': wasserstein_dist['Wasserstein Distance']
                }

    return metrics

def calculate_overall_average(all_metrics, categorical_columns, continuous_columns):
    combined_metrics = {
        'mean_difference': {},
        'variance_difference': {},
        'js_divergence': {},
        'wasserstein_dist': {},
        'overall_average': {}
    }

    if categorical_columns:
        for column in categorical_columns:
            if column in all_metrics[0]:
                combined_metrics['js_divergence'][column] = np.mean([m[column]['js_divergence'] for m in all_metrics])

    if continuous_columns:
        for column in continuous_columns:
            if column in all_metrics[0]:
                combined_metrics['mean_difference'][column] = {
                    'Average Real Mean': np.mean([m[column]['mean_difference']['Real Mean'] for m in all_metrics]),
                    'Average Fake Mean': np.mean([m[column]['mean_difference']['Fake Mean'] for m in all_metrics]),
                    'Average Mean Difference': np.mean([m[column]['mean_difference']['Mean Difference'] for m in all_metrics])
                }
                combined_metrics['variance_difference'][column] = {
                    'Average Real Variance': np.mean([m[column]['variance_difference']['Real Variance'] for m in all_metrics]),
                    'Average Fake Variance': np.mean([m[column]['variance_difference']['Fake Variance'] for m in all_metrics]),
                    'Average Variance Difference': np.mean([m[column]['variance_difference']['Variance Difference'] for m in all_metrics])
                }
                combined_metrics['wasserstein_dist'][column] = np.mean([m[column]['wasserstein_dist'] for m in all_metrics])
    
    # Calculate the overall average only if any metrics are present
    if combined_metrics['mean_difference'] or combined_metrics['js_divergence']:
        combined_metrics['overall_average'] = {
            'Average Mean Difference': np.mean([mean['Average Mean Difference'] for mean in combined_metrics['mean_difference'].values()]) if combined_metrics['mean_difference'] else None,
            'Average Variance Difference': np.mean([var['Average Variance Difference'] for var in combined_metrics['variance_difference'].values()]) if combined_metrics['variance_difference'] else None,
            'Average Jensen Shannon Divergence': np.mean(list(combined_metrics['js_divergence'].values())) if combined_metrics['js_divergence'] else None,
            'Average Wasserstein Distance': np.mean(list(combined_metrics['wasserstein_dist'].values())) if combined_metrics['wasserstein_dist'] else None
        }

    return combined_metrics

def evaluate_and_save_results(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, categorical_columns, continuous_columns):
    # Select up to 5 categorical and 5 continuous columns
    selected_categorical, selected_continuous = select_columns(categorical_columns, continuous_columns)

    all_metrics = []
    detailed_jsons = {}

    for i, fake_path in enumerate(fake_paths, 1):
        real_data, fake_data = load_data(real_path, fake_path)
        
        # Calculate metrics for the current fake dataset
        metrics = calculate_augmentation_metrics(real_data, fake_data, selected_categorical, selected_continuous)
        
        # Store the metrics for detailed results
        detailed_jsons[f"Augmented dataset {i}"] = metrics
        
        # Append metrics to list for calculating the overall average later
        all_metrics.append(metrics)

    # Calculate the overall average across all augmented datasets
    if all_metrics:
        overall_average = calculate_overall_average(all_metrics, selected_categorical, selected_continuous)
        # Add the overall average to the JSON
        detailed_jsons["Average of all datasets"] = overall_average
    
    # Save the combined detailed results and the overall average into one JSON file
    output_filename = os.path.join(performance_dir, f"{tds_model_name}_{dataset_name}_augmentation_evaluation.json")
    with open(output_filename, 'w') as f:
        json.dump(detailed_jsons, f, indent=4)

def augmentation_main(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, categorical_columns, continuous_columns):
    evaluate_and_save_results(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, categorical_columns, continuous_columns)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate augmentation quality in synthetic data")
    parser.add_argument("real_path", type=str, help="Path to the real data CSV file")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("tds_model_name", type=str, help="Name of the TDS model")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    parser.add_argument("categorical_columns", type=str, help="Comma-separated list of categorical columns")
    parser.add_argument("continuous_columns", type=str, help="Comma-separated list of continuous columns")
    args = parser.parse_args()

    categorical_columns = args.categorical_columns.split(',')
    continuous_columns = args.continuous_columns.split(',')

    # Find all fake dataset paths
    fake_paths = [os.path.join(args.performance_dir, f"{args.tds_model_name}_{args.dataset_name}_{i}.csv") for i in range(1, 6)]

    augmentation_main(args.real_path, fake_paths, args.dataset_name, args.tds_model_name, args.performance_dir, categorical_columns, continuous_columns)