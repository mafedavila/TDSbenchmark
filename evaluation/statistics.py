import numpy as np
import pandas as pd
import os
import json

def load_data(fake_path):
    """
    Load the synthetic dataset from a CSV file.
    """
    data = pd.read_csv(fake_path)
    return data

def compute_continuous_stats(data, columns):
    """
    Compute statistics for continuous columns.
    """
    stats = []
    for column in columns:
        if column not in data:
            continue
        col_data = data[column].dropna()
        stats.append({
            'column': column,
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'variance': float(col_data.var()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'std_dev': float(col_data.std()),
            '25th_percentile': float(col_data.quantile(0.25)),
            '50th_percentile': float(col_data.quantile(0.50)),
            '75th_percentile': float(col_data.quantile(0.75))
        })
    return stats

def compute_categorical_stats(data, columns):
    """
    Compute statistics for categorical columns.
    """
    stats = []
    for column in columns:
        if column not in data:
            continue
        col_data = data[column].dropna()
        mode = col_data.mode()
        mode_value = mode.iloc[0] if not mode.empty else np.nan
        stats.append({
            'column': column,
            'mode': str(mode_value),
            'unique_values': int(col_data.nunique())
        })
    return stats

def compute_average_stats(all_stats):
    """
    Compute the average statistics across multiple datasets for continuous columns.
    """
    avg_stats = {}
    count_stats = {}

    for stat in all_stats:
        col = stat['column']
        if col not in avg_stats:
            avg_stats[col] = {key: 0 for key in stat if key != 'column'}
            count_stats[col] = 0

        for key, value in stat.items():
            if key != 'column' and value is not None:
                avg_stats[col][key] += value
                count_stats[col] += 1

    for col, stats in avg_stats.items():
        for key in stats:
            avg_stats[col][key] /= count_stats[col]

    return avg_stats

def compute_average_categorical_stats(all_stats):
    """
    Compute the average statistics for categorical columns.
    """
    avg_stats = {}
    mode_counts = {}
    unique_value_list = {}

    for stat in all_stats:
        col = stat['column']
        if col not in avg_stats:
            mode_counts[col] = {}
            unique_value_list[col] = []

        # Count modes
        mode_value = stat.get('mode')
        if mode_value:
            if mode_value not in mode_counts[col]:
                mode_counts[col][mode_value] = 0
            mode_counts[col][mode_value] += 1

        # Collect unique values for each dataset
        unique_value_list[col].append(stat.get('unique_values', 0))

    # Calculate the most frequent mode and store unique values for each dataset
    avg_categorical_stats = {}
    for col in mode_counts:
        most_frequent_mode = max(mode_counts[col], key=mode_counts[col].get)
        avg_categorical_stats[col] = {
            'mode': most_frequent_mode,
            'unique_values_per_dataset': unique_value_list[col]  # Store the array of unique values
        }

    return avg_categorical_stats

def evaluate_all_datasets(dataset_name, tool_name, performance_dir, categorical_columns, continuous_columns):
    """
    Evaluate and save statistics for multiple synthetic datasets.
    """
    fake_datasets_dir = os.path.join('fake_datasets', tool_name)
    fake_paths = [os.path.join(fake_datasets_dir, f"{tool_name}_{dataset_name}_{i}.csv") for i in range(1, 6)]
    
    all_continuous_stats = []
    all_categorical_stats = []
    detailed_results = []

    for i, fake_path in enumerate(fake_paths, 1):
        try:
            # Load data
            fake_data = load_data(fake_path)

            # Compute statistics for the synthetic data
            continuous_stats = compute_continuous_stats(fake_data, continuous_columns)
            categorical_stats = compute_categorical_stats(fake_data, categorical_columns)

            # Combine continuous and categorical stats into one list
            dataset_stats = {'Fake dataset': i, 'continuous_stats': continuous_stats, 'categorical_stats': categorical_stats}
            detailed_results.append(dataset_stats)

            # Collect all stats for averaging later
            all_continuous_stats.extend(continuous_stats)
            all_categorical_stats.extend(categorical_stats)
        except Exception as e:
            pass
    
    # Save detailed statistics to a single JSON file
    detailed_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_statistics_evaluation_detailed.json")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_results, f, indent=4)

    # Compute average statistics across all datasets
    average_continuous_stats = compute_average_stats(all_continuous_stats)
    average_categorical_stats = compute_average_categorical_stats(all_categorical_stats)

    # Combine continuous and categorical average stats into one JSON
    average_stats = {
        'continuous_stats': average_continuous_stats,
        'categorical_stats': average_categorical_stats
    }

    # Save the average statistics to a JSON file
    avg_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_statistics_averages.json")
    with open(avg_output_filename, 'w') as f:
        json.dump(average_stats, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute and save statistical measures for synthetic datasets.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("tool_name", type=str, help="Name of the tool (TDS model)")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    parser.add_argument("categorical_columns", type=str, help="Comma-separated list of categorical columns")
    parser.add_argument("continuous_columns", type=str, help="Comma-separated list of continuous columns")
    args = parser.parse_args()

    # Convert comma-separated columns to lists
    categorical_columns = [col.strip() for col in args.categorical_columns.split(',') if col.strip()]
    continuous_columns = [col.strip() for col in args.continuous_columns.split(',') if col.strip()]

    evaluate_all_datasets(args.dataset_name, args.tool_name, args.performance_dir, categorical_columns, continuous_columns)
