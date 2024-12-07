import sys
import os
import pandas as pd
import json
from imblearn.over_sampling import SMOTE

def load_data(dataset_path):
    print("Loading dataset...", flush=True)
    return pd.read_csv(dataset_path)

def train_model(data):
    print("Applying SMOTE to generate synthetic data...", flush=True)
    smote = SMOTE(random_state=42)
    
    # Use the last column as the target
    target_column = data.columns[-1]
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Apply SMOTE to generate synthetic data
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Combine resampled features and target
    synthetic_data = pd.DataFrame(X_resampled, columns=X.columns)
    synthetic_data[target_column] = y_resampled

    return synthetic_data

def save_synthetic_data(synthetic_data, output_path, dataset_name, tool_name, version):
    output_file = os.path.join(output_path, f"{tool_name}_{dataset_name}_{version}.csv")
    print(f"Saving synthetic data to {output_file}...", flush=True)
    synthetic_data.to_csv(output_file, index=False)

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 run_tool.py <dataset_name> <output_path> <experiment_file>", flush=True)
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_path = os.path.join('..', '..', sys.argv[2])
    experiment_file = os.path.join('..', '..', sys.argv[3])

    with open(experiment_file, 'r') as file:
        experiments = json.load(file)
    
    experiment = next((exp for exp in experiments if exp['dataset'] == dataset_name), None)
    if not experiment:
        print(f"No experiment configuration found for dataset {dataset_name}", flush=True)
        sys.exit(1)
    
    tool_name = experiment.get('toolname', 'tool')

    print("Categorical columns:", experiment['columns'].get('categorical', []), flush=True)
    print("Continuous columns:", experiment['columns'].get('continuous', []), flush=True)

    dataset_path = os.path.join('..', '..', f"data/{dataset_name}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Load the dataset
        data = load_data(dataset_path)

        # Generate and save synthetic datasets 5 times
        for i in range(1, 6):
            # Generate synthetic data using SMOTE
            synthetic_data = train_model(data)

            # Save synthetic data with the correct toolname and dataset name (e.g., toolname_dataset_1.csv, toolname_dataset_2.csv)
            save_synthetic_data(synthetic_data, output_path, dataset_name, tool_name, i)

        print(f"5 synthetic datasets saved with names {tool_name}_{dataset_name}_1.csv to {tool_name}_{dataset_name}_5.csv", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

if __name__ == "__main__":
    main()