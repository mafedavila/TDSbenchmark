import sys
import os

# Set environment variables before importing any libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
import json
import autodiff.Codes.process_GQ as pce
import autodiff.Codes.autoencoder as ae
import autodiff.Codes.diffusion as diff
import autodiff.Codes.TabDDPMdiff as TabDiff

def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU is available: {gpu_name}", flush=True)
        return gpu_name
    else:
        print("No GPU available", flush=True)
        return None

def set_environment_variables(gpu_name):
    if (gpu_name and "RTX 40" in gpu_name):
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(dataset_path):
    print("Loading dataset...", flush=True)
    return pd.read_csv(dataset_path)

def train_model(data, dataset_path, experiment):
    print("Training model...", flush=True)
    experiment_dir = f"{dataset_path}_training"
    
    threshold = 0.01  # Threshold for mixed-type variables
    print("Step 1: Fitting DataFrameParser...", flush=True)
    parser = pce.DataFrameParser().fit(data, threshold)
    print("Step 1: Completed.", flush=True)

    device = 'cuda'
    n_epochs = 5000
    eps = 1e-5
    weight_decay = 1e-6
    maximum_learning_rate = 1e-2
    lr = 2e-4
    hidden_size = 250
    num_layers = 3
    batch_size = 50

    print("Step 2: Training autoencoder...", flush=True)
    ds = ae.train_autoencoder(data, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold)
    latent_features = ds[1].detach()
    print("Step 2: Completed.", flush=True)

    # diffusion hyper-parameters
    diff_n_epochs = 5000
    hidden_dims = (256, 512, 1024, 512, 256)
    converted_table_dim = latent_features.shape[1]
    sigma = 20
    num_batches_per_epoch = 50
    batch_size = 50
    T = 100

    print("Step 3: Training diffusion model...", flush=True)
    score = TabDiff.train_diffusion(latent_features, T, eps, sigma, lr, \
                        num_batches_per_epoch, maximum_learning_rate, weight_decay, diff_n_epochs, batch_size)
    print("Step 3: Completed.", flush=True)

    return score, ds, data, threshold

def sample_and_generate_output(score, ds, data, threshold, T, device):
    N = ds[1].shape[0]
    P = ds[1].shape[1]

    print("Sampling using Euler-Maruyama method...", flush=True)
    sample = diff.Euler_Maruyama_sampling(score, T, N, P, device)
    print("Sampling Completed.", flush=True)

    print("Generating output...", flush=True)
    gen_output = ds[0](sample, ds[2], ds[3])
    synthetic_data = pce.convert_to_table(data, gen_output, threshold)
    print("Output Generation Completed.", flush=True)

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

    gpu_name = check_gpu()
    set_environment_variables(gpu_name)

    dataset_path = os.path.join('..', '..', f"data/{dataset_name}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Load the dataset
        data = load_data(dataset_path)

        # Train the model once
        score, ds, data, threshold = train_model(data, dataset_path, experiment)

        # Generate and save synthetic datasets 5 times
        for i in range(1, 6):
            # Sample and generate synthetic data
            synthetic_data = sample_and_generate_output(score, ds, data, threshold, T=300, device='cuda')

            # Save synthetic data with the correct toolname and dataset name (e.g., toolname_dataset_1.csv, toolname_dataset_2.csv)
            save_synthetic_data(synthetic_data, output_path, dataset_name, tool_name, i)

        print(f"5 synthetic datasets saved with names {tool_name}_{dataset_name}_1.csv to {tool_name}_{dataset_name}_5.csv", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

if __name__ == "__main__":
    main()
