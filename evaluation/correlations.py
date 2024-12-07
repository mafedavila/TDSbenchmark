import pandas as pd
import numpy as np
import os
import json
from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr
import random

# Load the synthetic dataset
def load_data(fake_path):
    fake_data = pd.read_csv(fake_path)
    fake_data = fake_data.head(10000)
    return fake_data

# Compute correlation between two columns using the specified method
def compute_correlation(data, col1, col2, method):
    try:
        if method == 'pearson':
            corr, _ = pearsonr(data[col1], data[col2])
        elif method == 'spearman':
            corr, _ = spearmanr(data[col1], data[col2])
        elif method == 'kendall':
            corr, _ = kendalltau(data[col1], data[col2])
        elif method == 'pointbiserial':
            corr, _ = pointbiserialr(data[col1], data[col2])
        else:
            raise ValueError(f"Unknown method: {method}")
        if np.isnan(corr):
            corr = 0  # Set to 0 if correlation calculation results in NaN
    except Exception as e:
        print(f"Error computing {method} correlation between {col1} and {col2}: {e}")
        corr = 0  # If an error occurs, set the correlation to 0
    return corr

# Select random pairs of columns from the provided list
def select_random_pairs(columns, num_pairs=10):
    num_columns = len(columns)
    all_pairs = [(i, j) for i in range(num_columns) for j in range(i+1, num_columns)]
    random.seed(42)
    selected_pairs = random.sample(all_pairs, min(num_pairs, len(all_pairs)))
    return selected_pairs

# Evaluate correlations for a random selection of column pairs
def evaluate_correlations(data, columns, method, num_pairs=10):
    correlation_results = []
    pairs = select_random_pairs(columns, num_pairs)

    for (i, j) in pairs:
        col1, col2 = columns[i], columns[j]
        corr = compute_correlation(data, col1, col2, method)
        correlation_results.append({'column1': col1, 'column2': col2, 'method': method, 'correlation': corr})

    return correlation_results

# Main function to evaluate correlations for all synthetic datasets
def evaluate_all_datasets(dataset_name, tool_name, performance_dir, categorical_columns, continuous_columns):
    # Use the correct directory to read fake datasets
    fake_datasets_dir = os.path.join('fake_datasets', tool_name)
    fake_paths = [os.path.join(fake_datasets_dir, f"{tool_name}_{dataset_name}_{i}.csv") for i in range(1, 6)]
    
    all_correlation_results = []
    detailed_jsons = {}

    for i, fake_path in enumerate(fake_paths, 1):
        # Load the synthetic dataset
        fake_data = load_data(fake_path)

        correlation_results = []

        # Evaluate Pearson correlations for continuous variables
        correlation_results.extend(evaluate_correlations(fake_data, continuous_columns, method='pearson'))

        # Evaluate Spearman correlations for all variables
        correlation_results.extend(evaluate_correlations(fake_data, continuous_columns + categorical_columns, method='spearman'))

        # Evaluate Kendall correlations for all variables
        correlation_results.extend(evaluate_correlations(fake_data, continuous_columns + categorical_columns, method='kendall'))

        # Evaluate Point-Biserial correlations for binary categorical and continuous variables
        for col1 in categorical_columns:
            if len(fake_data[col1].unique()) == 2:  # binary columns only
                for col2 in continuous_columns:
                    corr = compute_correlation(fake_data, col1, col2, method='pointbiserial')
                    correlation_results.append({'column1': col1, 'column2': col2, 'method': 'pointbiserial', 'correlation': corr})

        # Add dataset results under a "Fake Dataset X" key
        detailed_jsons[f"Fake Dataset {i}"] = correlation_results
        all_correlation_results.extend(correlation_results)

    # Save all detailed results for all datasets into one JSON
    detailed_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_correlations_evaluation_detailed.json")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_jsons, f, indent=4)

    # Convert all results to a DataFrame for averaging
    combined_df = pd.DataFrame(all_correlation_results)

    # Calculate the average correlations per column pair and method
    avg_correlation_df = combined_df.groupby(['column1', 'column2', 'method'])['correlation'].mean().reset_index()

    # Convert the DataFrame to a dictionary format for the JSON
    avg_correlation_dict = avg_correlation_df.to_dict(orient='records')

    # Save the average correlations to a JSON file
    avg_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_correlations_averages.json")
    with open(avg_output_filename, 'w') as f:
        json.dump(avg_correlation_dict, f, indent=4)

    #print(f"Correlations averages saved to '{avg_output_filename}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute and save correlation matrices for synthetic datasets.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("tool_name", type=str, help="Name of the tool (TDS model)")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    parser.add_argument("categorical_columns", type=str, help="Comma-separated list of categorical columns")
    parser.add_argument("continuous_columns", type=str, help="Comma-separated list of continuous columns")
    args = parser.parse_args()

    # Convert the comma-separated strings to lists
    categorical_columns = [col.strip() for col in args.categorical_columns.split(',') if col.strip()]
    continuous_columns = [col.strip() for col in args.continuous_columns.split(',') if col.strip()]

    evaluate_all_datasets(args.dataset_name, args.tool_name, args.performance_dir, categorical_columns, continuous_columns)
