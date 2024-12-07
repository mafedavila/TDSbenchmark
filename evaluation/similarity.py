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

def total_variation_distance(real_data, fake_data, columns):
    tvd = {}
    for column in columns:
        if column not in real_data or column not in fake_data:
            continue
        p = real_data[column].value_counts(normalize=True, sort=False)
        q = fake_data[column].value_counts(normalize=True, sort=False)
        p, q = p.align(q, fill_value=0)
        tvd[column] = 0.5 * np.sum(np.abs(p - q))
    avg_tvd = np.mean(list(tvd.values())) if tvd else 0
    return tvd, avg_tvd

def skewness(fake_data, columns):
    skewness_vals = {}
    for column in columns:
        if column not in fake_data:
            continue
        skewness_vals[column] = skew(fake_data[column].dropna())
    avg_skewness = np.mean(list(skewness_vals.values())) if skewness_vals else 0
    return skewness_vals, avg_skewness

def jensen_shannon_divergence(real_data, fake_data, columns):
    js_divergence = {}
    for column in columns:
        if column not in real_data or column not in fake_data:
            continue
        p = real_data[column].value_counts(normalize=True, sort=False)
        q = fake_data[column].value_counts(normalize=True, sort=False)
        p, q = p.align(q, fill_value=0)
        js_divergence[column] = jensenshannon(p, q)
    js_values = [v for v in js_divergence.values() if np.isfinite(v)]
    avg_js_divergence = np.mean(js_values) if js_values else 0
    return js_divergence, avg_js_divergence

def wasserstein_distance_metric(real_data, fake_data, columns):
    wasserstein_dist = {}
    for column in columns:
        if column not in real_data or column not in fake_data:
            continue
        wasserstein_dist[column] = wasserstein_distance(real_data[column].dropna(), fake_data[column].dropna())
    w_dist_values = [v for v in wasserstein_dist.values() if np.isfinite(v)]
    avg_wasserstein_dist = np.mean(w_dist_values) if w_dist_values else 0
    return wasserstein_dist, avg_wasserstein_dist

def calculate_similarity_metrics(real_data, fake_data, categorical_columns, continuous_columns):
    tvd, avg_tvd = total_variation_distance(real_data, fake_data, categorical_columns)
    skewness_vals, avg_skewness = skewness(fake_data, continuous_columns)
    js_divergence, avg_js_divergence = jensen_shannon_divergence(real_data, fake_data, categorical_columns)
    wasserstein_dist, avg_wasserstein_dist = wasserstein_distance_metric(real_data, fake_data, continuous_columns)

    # Create a dictionary of average metrics
    average_metrics = {
        'Average Total Variation Distance': avg_tvd,
        'Average Skewness': avg_skewness,
        'Average Jensen Shannon Divergence': avg_js_divergence,
        'Average Wasserstein Distance': avg_wasserstein_dist,
    }
    
    return {
        'tvd': tvd, 'skewness_vals': skewness_vals, 'js_divergence': js_divergence,
        'wasserstein_dist': wasserstein_dist, 'average_metrics': average_metrics
    }

def evaluate_and_save_results(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, categorical_columns, continuous_columns):
    all_metrics = []
    detailed_jsons = []

    for i, fake_path in enumerate(fake_paths, 1):
        real_data, fake_data = load_data(real_path, fake_path)
        
        # Calculate metrics for the current fake dataset
        metrics = calculate_similarity_metrics(real_data, fake_data, categorical_columns, continuous_columns)
        
        # Append metrics for detailed results
        detailed_jsons.append({f"Fake dataset {i}": metrics})
        
        # Store the average metrics for later processing
        all_metrics.append(metrics['average_metrics'])

    # Save the combined detailed results into one JSON file
    detailed_output_filename = os.path.join(performance_dir, f"{tds_model_name}_{dataset_name}_similarity_evaluation_detailed.json")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_jsons, f, indent=4)

    # Save the average metrics for each dataset
    avg_metrics_df = pd.DataFrame(all_metrics)
    avg_output_filename = os.path.join(performance_dir, f"{tds_model_name}_{dataset_name}_similarity_averages_per_dataset.csv")
    avg_metrics_df.to_csv(avg_output_filename, index=False)

    # Calculate and save the overall averages
    overall_avg = avg_metrics_df.mean().to_dict()
    overall_avg_df = pd.DataFrame([overall_avg])
    overall_avg_output_filename = os.path.join(performance_dir, f"{tds_model_name}_{dataset_name}_similarity_overall_average.csv")
    overall_avg_df.to_csv(overall_avg_output_filename, index=False)

def main(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, categorical_columns, continuous_columns):
    evaluate_and_save_results(real_path, fake_paths, dataset_name, tds_model_name, performance_dir, categorical_columns, continuous_columns)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate synthetic data quality")
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

    main(args.real_path, fake_paths, args.dataset_name, args.tds_model_name, args.performance_dir, categorical_columns, continuous_columns)
