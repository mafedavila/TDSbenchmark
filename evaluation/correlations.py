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

# Calculate higher-order cross moments and cumulants
def calculate_cross_cumulants(df, columns):

    results = []

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            col1 = columns[i]
            col2 = columns[j]

            pair = df[[col1, col2]].dropna()

            if len(pair) < 2:
                continue

            x = pair[col1]
            y = pair[col2]

            # Standardize variables
            zx = (x - x.mean()) / x.std()
            zy = (y - y.mean()) / y.std()

            # Pearson correlation (second-order dependency)
            pearson = np.mean(zx * zy)


            # Higher-order mixed moments
            M21 = np.mean(zx**2 * zy)
            M12 = np.mean(zx * zy**2)

            M31 = np.mean(zx**3 * zy)
            M22 = np.mean(zx**2 * zy**2)
            M13 = np.mean(zx * zy**3)


            # Cross cumulants
            K21 = M21
            K12 = M12

            K31 = M31 - 3 * pearson

            K22 = M22 - 1 - 2 * (pearson ** 2)

            K13 = M13 - 3 * pearson


            results.append(
                {
                    "column1": col1,
                    "column2": col2,
                    "pearson": pearson,

                    # Third order
                    "K21": K21,
                    "K12": K12,

                    # Fourth order
                    "K31": K31,
                    "K22": K22,
                    "K13": K13
                }
            )

    return results

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

    fake_paths = [
        os.path.join(
            fake_datasets_dir,
            f"{tool_name}_{dataset_name}_{i}.csv"
        )
        for i in range(1, 6)
    ]

    all_correlation_results = []
    all_cumulant_results = []

    detailed_jsons = {}


    for i, fake_path in enumerate(fake_paths, 1):

        # Load synthetic dataset
        fake_data = load_data(fake_path)

        correlation_results = []

        # Pearson correlations for continuous variables
        correlation_results.extend(
            evaluate_correlations(
                fake_data,
                continuous_columns,
                method='pearson'
            )
        )


        # Spearman correlations
        correlation_results.extend(
            evaluate_correlations(
                fake_data,
                continuous_columns + categorical_columns,
                method='spearman'
            )
        )


        # Kendall correlations
        correlation_results.extend(
            evaluate_correlations(
                fake_data,
                continuous_columns + categorical_columns,
                method='kendall'
            )
        )


        # Point-biserial correlations
        for col1 in categorical_columns:

            if len(fake_data[col1].unique()) == 2:

                for col2 in continuous_columns:

                    corr = compute_correlation(
                        fake_data,
                        col1,
                        col2,
                        method='pointbiserial'
                    )

                    correlation_results.append(
                        {
                            'column1': col1,
                            'column2': col2,
                            'method': 'pointbiserial',
                            'correlation': corr
                        }
                    )


        # Store correlation results
        detailed_jsons[f"Fake Dataset {i}"] = correlation_results

        all_correlation_results.extend(
            correlation_results
        )

        print(
            f"Evaluating cumulants for Fake Dataset {i}"
        )

        cumulant_results = calculate_cross_cumulants(
            fake_data,
            continuous_columns
        )

        all_cumulant_results.extend(
            cumulant_results
        )

    detailed_output_filename = os.path.join(
        performance_dir,
        f"{tool_name}_{dataset_name}_correlations_evaluation_detailed.json"
    )

    with open(detailed_output_filename, 'w') as f:

        json.dump(
            detailed_jsons,
            f,
            indent=4
        )

    cumulant_output_filename = os.path.join(
        performance_dir,
        f"{tool_name}_{dataset_name}_cumulants_detailed.json"
    )

    with open(cumulant_output_filename, 'w') as f:

        json.dump(
            all_cumulant_results,
            f,
            indent=4
        )

    combined_df = pd.DataFrame(
        all_correlation_results
    )


    avg_correlation_df = (
        combined_df
        .groupby(
            [
                'column1',
                'column2',
                'method'
            ]
        )['correlation']
        .mean()
        .reset_index()
    )


    avg_correlation_filename = os.path.join(
        performance_dir,
        f"{tool_name}_{dataset_name}_correlations_averages.json"
    )


    with open(avg_correlation_filename, 'w') as f:

        json.dump(
            avg_correlation_df.to_dict(
                orient='records'
            ),
            f,
            indent=4
        )

    cumulant_df = pd.DataFrame(
        all_cumulant_results
    )


    if not cumulant_df.empty:

        avg_cumulant_df = (
            cumulant_df
            .groupby(
                [
                    'column1',
                    'column2'
                ]
            )
            .mean()
            .reset_index()
        )


        avg_cumulant_filename = os.path.join(
            performance_dir,
            f"{tool_name}_{dataset_name}_cumulants_averages.json"
        )


        with open(avg_cumulant_filename, 'w') as f:

            json.dump(
                avg_cumulant_df.to_dict(
                    orient='records'
                ),
                f,
                indent=4
            )

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
