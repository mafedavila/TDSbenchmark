import sys
import os
import pandas as pd
import numpy as np
import subprocess
import json
import torch
from sklearn.preprocessing import LabelEncoder

def check_gpu():
    # Not used in tabsyn implementation
    pass

def set_environment_variables(gpu_name):
    # Not used in tabsyn implementation
    pass

def load_data(dataset_path):
    # Not used in tabsyn implementation
    pass

def train_model(dataset_name):
    print("Training model using tabsyn...", flush=True)
    # Run the training script for tabddpm
    print(dataset_name)

    subprocess.run(['python', os.path.join('main.py'), '--dataname', dataset_name, '--method', 'vae', '--mode', 'train'], check=True)
    subprocess.run(['python', os.path.join('main.py'), '--dataname', dataset_name, '--method', 'tabsyn', '--mode', 'train'], check=True)
    

def generate_synthetic_data(dataset_name, tool_name):
    # Check if folder synthetic/dataset_name exists, if not create it
    folder_path = os.path.join('synthetic', dataset_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Run the subprocess to generate the synthetic data
    subprocess.run(['python', os.path.join('main.py'), '--dataname', dataset_name, '--method', 'tabsyn', '--mode', 'sample'], check=True)

    # Read the generated synthetic data into a pandas DataFrame
    synthetic_data = pd.read_csv(os.path.join(folder_path, f'{tool_name}.csv'))

    return synthetic_data

def save_synthetic_data(dataset_name, tool_name, version, synthetic_data, output_path):
    output_file = os.path.join(output_path, f"{tool_name}_{dataset_name}_{version}.csv")
    print(f"Saving synthetic data to {output_file}...", flush=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Save encoded data
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

    try:
        # Train the model
        train_model(dataset_name)

        # Generate and save synthetic datasets 5 times
        for i in range(1, 6):
            # Generate synthetic data
            synthetic_data = generate_synthetic_data(dataset_name, tool_name)

            # Save synthetic data with the correct toolname and dataset name (e.g., toolname_dataset_1.csv, toolname_dataset_2.csv)
            save_synthetic_data(dataset_name, tool_name, i, synthetic_data, output_path)

        print(f"5 synthetic datasets saved with names {tool_name}_{dataset_name}_1.csv to {tool_name}_{dataset_name}_5.csv", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

if __name__ == "__main__":
    main()
