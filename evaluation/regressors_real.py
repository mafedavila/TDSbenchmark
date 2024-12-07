import os
import pandas as pd
import numpy as np
from math import nan
import json

from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Function to evaluate each regressor
def evaluate_regressor(name, model, X_train, X_test, y_train, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        evs = explained_variance_score(y_test, y_pred)
    except ValueError:
        evs = nan

    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except ValueError:
        mape = nan

    try:
        r2s = r2_score(y_test, y_pred)
    except ValueError:
        r2s = nan

    results.append({
        "Regression Model": name,
        "Explained Variance Score": evs,
        "Mean Abs Percentage Error": mape,
        "R2 Score": r2s
    })

# Function to evaluate regressors on a single dataset
def regression_evaluation(path, predicted, dataset, ds_model):
    my_df = pd.read_csv(path)
    my_df = my_df.head(10000)

    y = my_df[predicted]
    drop_elements = [predicted]
    X = my_df.drop(drop_elements, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of regressors to evaluate
    regressors = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge()),
        ("Lasso Regression", Lasso()),
        ("Bayesian Ridge Regression", BayesianRidge())
    ]

    results = []
    for name, model in regressors:
        evaluate_regressor(name, model, X_train, X_test, y_train, y_test, results)

    return results

# Function to evaluate all fake datasets and calculate averages
def evaluate_all_datasets(predicted, dataset_name, ds_model, performance_dir):
    # Correct directory to search for datasets
    fake_datasets_dir = os.path.join('fake_datasets', ds_model)
    
    # Find all the files that match the pattern toolname_dataset_n.csv (where n is 1, 2, 3...)
    fake_files = sorted([f for f in os.listdir(fake_datasets_dir) 
                         if f.startswith(f"{ds_model}_{dataset_name}_") and f.endswith('.csv')])

    fake_paths = [os.path.join(fake_datasets_dir, f) for f in fake_files]

    if not fake_paths:
        print(f"No datasets found for {ds_model} and {dataset_name} in {fake_datasets_dir}")
        return

    all_results = []
    detailed_results = {}  # Dictionary to store all detailed results

    # Evaluate each fake dataset
    for i, fake_path in enumerate(fake_paths, 1):
        results = regression_evaluation(fake_path, predicted, f"{dataset_name}_{i}", ds_model)
        detailed_results[f"Fake dataset {i}"] = results

        # Collect results to compute averages
        all_results.append(results)

    # Save all detailed results to a single JSON file
    detailed_output_filename = os.path.join(performance_dir, f"{ds_model}_{dataset_name}_regression_evaluation_detailed.json")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_results, f, indent=4)

    # Combine all results into a DataFrame for easy averaging
    combined_results = []
    for result_set in all_results:
        for result in result_set:
            combined_results.append(result)

    combined_df = pd.DataFrame(combined_results)

    # Calculate the average for each regressor across the datasets
    average_results = combined_df.groupby("Regression Model").mean().reset_index()

    # Save the average metrics to a CSV file
    avg_output_filename = os.path.join(performance_dir, f"{ds_model}_{dataset_name}_regression_evaluation_average.csv")
    average_results.to_csv(avg_output_filename, index=False)

if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    import json

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate regressors on real datasets")
    parser.add_argument("real_dataset_path", type=str, help="Path to the real dataset CSV")
    parser.add_argument("predicted", type=str, help="The target column to predict")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    args = parser.parse_args()

    # Extract the dataset name from the real dataset path
    dataset_name = os.path.splitext(os.path.basename(args.real_dataset_path))[0]  # Get file name without extension

    # Evaluate the real dataset
    real_results = regression_evaluation(args.real_dataset_path, args.predicted, dataset_name, "real")

    # Save detailed results for the real dataset
    real_detailed_output = os.path.join(
        args.performance_dir, f"real_{dataset_name}_regression_evaluation_detailed.json"
    )
    with open(real_detailed_output, 'w') as f:
        json.dump({"Real Dataset": real_results}, f, indent=4)

    # Combine results into a DataFrame
    real_df = pd.DataFrame(real_results)

    # Save average metrics for the real dataset
    real_avg_output = os.path.join(
        args.performance_dir, f"real_{dataset_name}_regression_evaluation_average.csv"
    )
    real_df.to_csv(real_avg_output, index=False)

    print(f"Real dataset evaluation results saved to:")
    print(f" - Detailed JSON: {real_detailed_output}")
    print(f" - Average CSV: {real_avg_output}")