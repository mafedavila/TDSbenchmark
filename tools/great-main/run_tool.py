import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
import json
from great.be_great import GReaT

def check_gpu():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}", flush=True)
            if "RTX 40" in gpu_name:  
                return i  
        print("No RTX 4090 found, using default GPU settings.", flush=True)
        return None
    else:
        print("No GPU available", flush=True)
        return None


def set_environment_variables(gpu_index):
    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)  # Only the RTX 4090 will be used
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        print(f"Using GPU {gpu_index} (RTX 4090) for training.", flush=True)
    else:
        print("Using default GPU configuration.", flush=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(dataset_path):
    print("Loading dataset...", flush=True)
    return pd.read_csv(dataset_path)

def train_model(data, dataset_path, experiment):
    print("Training model...", flush=True)
    experiment_dir = f"{dataset_path}_training"
    
    model = GReaT("distilgpt2",                         
              epochs=50,                             
              save_steps=10000,                      
              logging_steps=1000,                    
              experiment_dir="trainer",      
              lr_scheduler_type="constant_with_warmup",        
              learning_rate=5e-4                   
             )

    trainer = model.fit(data)

    return model

def generate_synthetic_data(model, data):
    print("Generating synthetic data...", flush=True)
    synthetic_data = model.sample(5000, k=50)
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

    # Print columns from the experiment configuration
    print("Categorical columns:", experiment['columns'].get('categorical', []), flush=True)
    print("Continuous columns:", experiment['columns'].get('continuous', []), flush=True)
    print("Integer columns:", experiment['columns'].get('integer', []), flush=True)
    print("Mixed columns:", experiment['columns'].get('mixed', {}), flush=True)
    print("Log columns:", experiment['columns'].get('log', []), flush=True)
    print("General columns:", experiment['columns'].get('general', []), flush=True)
    print("Non-categorical columns:", experiment['columns'].get('non_categorical', []), flush=True)

    
    gpu_index = check_gpu()  
    set_environment_variables(gpu_index)  

    dataset_path = os.path.join('..', '..', f"data/{dataset_name}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Load the dataset
        data = load_data(dataset_path)

        # Train the model
        model = train_model(data, dataset_path, experiment)

        # Generate and save synthetic datasets 5 times
        for i in range(1, 6):
            # Generate synthetic data
            synthetic_data = generate_synthetic_data(model, data)

            # Save synthetic data with the correct toolname and dataset name (e.g., toolname_dataset_1.csv, toolname_dataset_2.csv)
            save_synthetic_data(synthetic_data, output_path, dataset_name, tool_name, i)

        print(f"5 synthetic datasets saved with names {tool_name}_{dataset_name}_1.csv to {tool_name}_{dataset_name}_5.csv", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

if __name__ == "__main__":
    main()

