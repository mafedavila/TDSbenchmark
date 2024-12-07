import sys
import os
import pandas as pd
import numpy as np
import subprocess
import json
from sklearn.preprocessing import LabelEncoder

def check_gpu():
    # Not used in tabddpm implementation
    pass

def set_environment_variables(gpu_name):
    # Not used in tabddpm implementation
    pass

def load_data(dataset_path):
    # Not used in tabddpm implementation
    pass

def train_model(config_path):
    print("Training model using tabddpm...", flush=True)
    # Run the training script for tabddpm
    print(config_path)
    subprocess.run(['python', os.path.join('scripts', 'pipeline.py'), '--config', config_path, '--train', '--sample', '--eval'], check=True)

def generate_synthetic_data(fake_dir):
    #print("Loading synthetic data...", flush=True)
    
    data_parts = []
    
    # Check if X_cat_train.npy exists and load if present
    cat_file_path = os.path.join(fake_dir, "X_cat_train.npy")
    if os.path.exists(cat_file_path):
        #print(f"Loading {cat_file_path}...", flush=True)
        X_cat_train = np.load(cat_file_path, allow_pickle=True)
        data_parts.append(X_cat_train)
    else:
        print(f"{cat_file_path} not found, skipping categorical data...", flush=True)
    
    # Check if X_num_train.npy exists and load if present
    num_file_path = os.path.join(fake_dir, "X_num_train.npy")
    if os.path.exists(num_file_path):
        #print(f"Loading {num_file_path}...", flush=True)
        X_num_train = np.load(num_file_path, allow_pickle=True)
        data_parts.append(X_num_train)
    else:
        print(f"{num_file_path} not found, skipping numerical data...", flush=True)
    
    # Check if y_train.npy exists and load if present
    y_file_path = os.path.join(fake_dir, "y_train.npy")
    if os.path.exists(y_file_path):
        #print(f"Loading {y_file_path}...", flush=True)
        y_train = np.load(y_file_path, allow_pickle=True)
    else:
        #print(f"{y_file_path} not found, skipping target data...", flush=True)
        y_train = np.array([])  # Default empty array for target if not found
    
    if not data_parts:
        raise ValueError("No valid data files found to generate synthetic dataset.")
    
    # Concatenate available data parts
    combined_data = np.concatenate(data_parts, axis=1) if len(data_parts) > 1 else data_parts[0]
    
    df_train = pd.DataFrame(data=combined_data)
    if y_train.size > 0:  # Only add target column if y_train was successfully loaded
        df_train['target'] = y_train

    return df_train

def encode_data(df_train):
    print("Encoding data...", flush=True)
    label_encoders = {}
    categorical_cols = df_train.select_dtypes(include=['object']).columns

    df_encoded = df_train.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    return df_train, df_encoded, label_encoders

def save_synthetic_data(df_encoded, output_path, dataset_name, tool_name, version):
    output_file = os.path.join(output_path, f"{tool_name}_{dataset_name}_{version}.csv")
    print(f"Saving synthetic data to {output_file}...", flush=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Save encoded data
    df_encoded.to_csv(output_file, index=False)


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
    config_path = os.path.join('exp', dataset_name, 'ddpm_cb_best', 'config.toml')
    fake_dir = os.path.join('exp', dataset_name, 'ddpm_cb_best')

    try:
        # Train the model
        train_model(config_path)

        # Generate and save synthetic datasets 5 times
        for i in range(1, 6):
            # Generate synthetic data
            synthetic_data = generate_synthetic_data(fake_dir)

            # Encode synthetic data
            df_train, df_encoded, label_encoders = encode_data(synthetic_data)

            # Save synthetic data with the correct toolname and dataset name (e.g., toolname_dataset_1.csv, toolname_dataset_2.csv)
            save_synthetic_data(df_encoded, output_path, dataset_name, tool_name, i)

        print(f"5 synthetic datasets saved with names {tool_name}_{dataset_name}_1.csv to {tool_name}_{dataset_name}_5.csv", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

if __name__ == "__main__":
    main()
