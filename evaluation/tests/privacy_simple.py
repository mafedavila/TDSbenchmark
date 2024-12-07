"""
Privacy Test for Synthetic vs. Real Data - Simple Privacy Evaluation

This script evaluates the privacy of synthetic datasets by comparing them to real datasets using density and detection accuracy metrics. 
It measures the distance between real and generated data using various privacy metrics including DCR (Distance to Closest Record), 
NNDR (Nearest Neighbor Distance Ratio), Correct Attribution Probability, k-Anonymity, k-Map, Delta-Presence, and Identifiability Score.

Good privacy protection is indicated by low detection accuracy, small density deviations, and favorable privacy metrics, 
suggesting that the synthetic data preserves privacy without compromising data utility.

### DCR Analysis and Interpretation

DCR is a critical metric that measures how close each synthetic record is to the nearest real record. The interpretation of DCR values is as follows:

- **Low DCR Values**: 
  - **Mean/Median**: Suggest that synthetic records are very close to real records, potentially compromising privacy as it may lead to re-identification risks.
  - **Standard Deviation**: A low standard deviation means that most synthetic records are similarly close to real records, indicating consistent privacy risks.
  - **Percentiles**: Low percentiles (e.g., 25th percentile) indicate that a significant portion of the synthetic data closely resembles real data, raising privacy concerns.

- **High DCR Values**: 
  - **Mean/Median**: Indicate that synthetic records are generally far from real records, which is favorable for privacy as it reduces the risk of re-identification.
  - **Standard Deviation**: A high standard deviation suggests variability in how close synthetic records are to real records, potentially balancing privacy and utility.
  - **Percentiles**: High percentiles suggest that most synthetic records are sufficiently distinct from real records, enhancing privacy protection.

### Usage:

1. **Direct Execution**:
   - Run the script from the command line with the required arguments, for example:
     ```
     python privacy_simple.py real_data.csv dataset_name tool_name output_dir
     ```
   - This will generate JSON files with detailed metrics for each synthetic dataset and an overall average.

2. **Importing in Another Script**:
   - Import the `evaluate_all_datasets` function:
     ```
     from privacy_simple import evaluate_all_datasets
     ```
   - Call the function with appropriate arguments:
     ```
     evaluate_all_datasets(real_path, dataset_name, tool_name, performance_dir)
     ```
   - `real_path`: Path to the real dataset CSV file.
   - `dataset_name`: Name of the dataset.
   - `tool_name`: Name of the synthetic data generation tool.
   - `performance_dir`: Directory where the output JSON files will be saved.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import os
import json

def load_data(real_path, fake_path):
    real_data = pd.read_csv(real_path)
    real_data = real_data.head(10000)
    fake_data = pd.read_csv(fake_path)
    fake_data = fake_data.head(10000)
    return real_data, fake_data

def aggregate_metric(values):
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std_dev': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        '25th_percentile': np.percentile(values, 25),
        '50th_percentile': np.percentile(values, 50),
        '75th_percentile': np.percentile(values, 75)
    }

def dcr(real_data, fake_data):
    dists = cdist(fake_data, real_data)
    min_dists = np.min(dists, axis=1)
    return aggregate_metric(min_dists)

def nndr(real_data, fake_data):
    dists = cdist(fake_data, real_data)
    min_dists = np.partition(dists, 1, axis=1)[:, :2]
    nndr_values = min_dists[:, 0] / min_dists[:, 1]
    return aggregate_metric(nndr_values)

def correct_attribution_probability(real_data, fake_data):
    dists = cdist(fake_data, real_data)
    closest_indices = np.argmin(dists, axis=1)
    attribution_counts = np.bincount(closest_indices, minlength=len(real_data))
    cap = np.max(attribution_counts) / len(fake_data)
    return cap

def k_anonymity(real_data, fake_data):
    combined_data = pd.concat([real_data, fake_data])
    group_sizes = combined_data.groupby(list(combined_data.columns)).size()
    k = group_sizes.min()
    return k

def k_map(real_data, fake_data):
    group_sizes = real_data.groupby(list(real_data.columns)).size()
    k_map_values = group_sizes.value_counts().min()
    return k_map_values

def delta_presence(real_data, fake_data):
    combined_data = pd.concat([real_data, fake_data])
    group_sizes_real = real_data.groupby(list(real_data.columns)).size()
    group_sizes_combined = combined_data.groupby(list(combined_data.columns)).size()
    delta_presence = (group_sizes_real / group_sizes_combined).max()
    return delta_presence

def identifiability_score(real_data, fake_data):
    dists = cdist(fake_data, real_data)
    closest_distances = np.min(dists, axis=1)
    identifiability_score_value = np.mean(closest_distances < np.median(closest_distances))
    return identifiability_score_value

def evaluate_all_datasets(real_path, dataset_name, tool_name, performance_dir):
    fake_datasets_dir = os.path.join('fake_datasets', tool_name)
    fake_paths = [os.path.join(fake_datasets_dir, f"{tool_name}_{dataset_name}_{i}.csv") for i in range(1, 6)]
    
    all_results = []
    detailed_jsons = []

    for i, fake_path in enumerate(fake_paths, 1):
        real_data, fake_data = load_data(real_path, fake_path)

        dcr_stats = dcr(real_data, fake_data)
        nndr_stats = nndr(real_data, fake_data)
        cap_value = correct_attribution_probability(real_data, fake_data)
        k_anonymity_value = k_anonymity(real_data, fake_data)
        k_map_value = k_map(real_data, fake_data)
        delta_presence_value = delta_presence(real_data, fake_data)
        identifiability_score_value = identifiability_score(real_data, fake_data)

        detailed_metrics = {
            'Dataset': f"{dataset_name}_{i}",
            'TDS Model': tool_name,
            'DCR': dcr_stats,
            'NNDR': nndr_stats,
            'Correct Attribution Probability': cap_value,
            'k-Anonymity': k_anonymity_value,
            'k-Map': k_map_value,
            'Delta-Presence': delta_presence_value,
            'Identifiability Score': identifiability_score_value
        }
        detailed_jsons.append(detailed_metrics)

    detailed_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_privacy_evaluation_detailed.json")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_jsons, f, indent=4)

    avg_metrics = {
        'Dataset': f"{dataset_name}_average",
        'TDS Model': tool_name,
        'DCR': aggregate_metric([res['DCR']['mean'] for res in detailed_jsons]),
        'NNDR': aggregate_metric([res['NNDR']['mean'] for res in detailed_jsons]),
        'Correct Attribution Probability': np.mean([res['Correct Attribution Probability'] for res in detailed_jsons]),
        'k-Anonymity': np.mean([res['k-Anonymity'] for res in detailed_jsons]),
        'k-Map': np.mean([res['k-Map'] for res in detailed_jsons]),
        'Delta-Presence': np.mean([res['Delta-Presence'] for res in detailed_jsons]),
        'Identifiability Score': np.mean([res['Identifiability Score'] for res in detailed_jsons])
    }

    avg_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_privacy_averages.json")
    with open(avg_output_filename, 'w') as f:
        json.dump(avg_metrics, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate privacy metrics for synthetic datasets.")
    parser.add_argument("real_path", type=str, help="Path to the real data CSV file")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("tool_name", type=str, help="Name of the TDS model")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    args = parser.parse_args()

    evaluate_all_datasets(args.real_path, args.dataset_name, args.tool_name, args.performance_dir)