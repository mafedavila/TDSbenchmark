import os
import subprocess
import pandas as pd
import json
import evaluation.classifiers as classifiers
import evaluation.regressors as regressors
from evaluation.similarity import main as similarity_main
from evaluation.statistics import evaluate_all_datasets as statistics_main
from evaluation.detection import evaluate_all_datasets as detection_main
from evaluation.privacy import evaluate_all_datasets as privacy_main
from evaluation.correlations import evaluate_all_datasets as correlations_main

# Function to create and activate a conda environment
def create_and_activate_conda_env(tool_name, tool_path):
    print(f"Starting environment setup for tool: {tool_name}")
    env_name = tool_name

    # Step 1: Read the Python version from the file
    python_version_file = os.path.join(tool_path, 'python_version.txt')
    if os.path.exists(python_version_file):
        with open(python_version_file, 'r') as file:
            python_version = file.read().strip()
    else:
        python_version = '3.9'  # Default to Python 3.9
        print(f"No python_version.txt found, using default Python version: {python_version}")

    # Step 2: Check if the environment already exists
    
    env_list = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
    
    env_exists = False
    for line in env_list.stdout.splitlines():
        if line.startswith(env_name + ' '):
            env_exists = True
            break

    if env_exists:
        print(f"Environment '{env_name}' already exists. Skipping creation.")
    else:
        print(f"Environment '{env_name}' does not exist. Proceeding to create it...")
        create_env_cmd = ['conda', 'create', '--name', env_name, '--yes', f'python={python_version}']
        result = subprocess.run(create_env_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to create conda environment '{env_name}'. Error: {result.stderr}")
        else:
            print(f"Conda environment '{env_name}' created successfully.")

    # Step 3: Get Conda base path and set environment variables
    print("Fetching Conda base path...")
    conda_prefix = subprocess.run(['conda', 'info', '--base'], capture_output=True, text=True).stdout.strip()
    env_path = os.path.join(conda_prefix, 'envs', env_name)
    
    # Set environment variables
    os.environ['PATH'] = os.path.join(env_path, 'bin') + os.pathsep + os.environ['PATH']
    os.environ['CONDA_PREFIX'] = env_path
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Step 4: Install requirements
    print("Installing requirements from requirements.txt...")
    requirements_path = os.path.join(tool_path, 'requirements.txt')
    if os.path.exists(requirements_path):
        install_cmd = f"source activate {env_name} && pip install -r {requirements_path}"
        result = subprocess.run(install_cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to install requirements for '{env_name}'. Error: {result.stderr}")
        else:
            print(f"Requirements installed successfully for environment '{env_name}'.")
    else:
        print(f"requirements.txt not found at {requirements_path}. Skipping package installation.")

    print(f"Conda environment setup completed for tool: {tool_name}")

# Function to start the monitoring script
def start_monitoring():
    print("Starting monitoring script...")
    monitor_script = './monitor_usageMac.sh'
    monitor_process = subprocess.Popen(['bash', monitor_script])
    return monitor_process

# Function to stop the monitoring script
def stop_monitoring(monitor_process):
    print("Stopping monitoring script...")
    monitor_process.terminate()
    monitor_process.wait()

# Function to load the dataset
def load_dataset(dataset_path):
    print(f"Loading dataset from {dataset_path}...")
    return pd.read_csv(dataset_path)

# Function to run the tool (assumes a specific entry point script for each tool)
def run_tool(tool_path, dataset, output_path, experiment_file):
    print(f"Running tool at {tool_path} on dataset {dataset}...")
    os.chdir(tool_path)
    tool_script = 'run_tool.py'
    subprocess.run(['python', tool_script, dataset, output_path, experiment_file], check=True)
    os.chdir('../../')

# Function to read CPU and memory performance metrics from the monitoring output
def read_cpu_mem_performance_metrics():
    print("Reading CPU and memory performance metrics...")
    performance_df = pd.read_csv("cpu_mem_usage.csv")
    mean_cpu = performance_df['CPU Usage (%)'].mean()
    max_cpu = performance_df['CPU Usage (%)'].max()
    std_cpu = performance_df['CPU Usage (%)'].std()

    mean_mem = performance_df['Memory Usage (%)'].mean()
    max_mem = performance_df['Memory Usage (%)'].max()
    std_mem = performance_df['Memory Usage (%)'].std()

    total_time = (performance_df.index[-1] - performance_df.index[0] + 1)  # Assuming 1-second intervals

    metrics = {
        'Mean CPU Usage (%)': mean_cpu,
        'Max CPU Usage (%)': max_cpu,
        'Std CPU Usage (%)': std_cpu,
        'Mean Memory Usage (%)': mean_mem,
        'Max Memory Usage (%)': max_mem,
        'Std Memory Usage (%)': std_mem,
        'Total Time (s)': total_time
    }
    return metrics

# Function to read GPU performance metrics from the monitoring output
def read_gpu_performance_metrics():
    print("Reading GPU performance metrics...")
    gpu_df = pd.read_csv("gpu_usage.csv")

    gpu_metrics = []
    for gpu_number in gpu_df['GPU Number'].unique():
        gpu_data = gpu_df[gpu_df['GPU Number'] == gpu_number]['GPU Usage (%)']
        mean_gpu = gpu_data.mean()
        max_gpu = gpu_data.max()
        std_gpu = gpu_data.std()

        gpu_metrics.append({
            'GPU Number': gpu_number,
            'Mean GPU Usage (%)': mean_gpu,
            'Max GPU Usage (%)': max_gpu,
            'Std GPU Usage (%)': std_gpu
        })

    return pd.DataFrame(gpu_metrics)

# Function to evaluate the generated datasets
def evaluate_synthetic_data(tool, dataset, performance_dir, task_type, target_column, categorical_columns, continuous_columns, subsample_size=1000):
    print(f"Evaluating synthetic data for {tool} on dataset {dataset}...")

    # Find all generated fake datasets
    fake_datasets = sorted([f for f in os.listdir(f'fake_datasets/{tool}') if f.startswith(f'{tool}_{dataset}_') and f.endswith('.csv')])
    if not fake_datasets:
        print(f"No datasets found for {tool} and {dataset}")
        return

    # Create fake dataset paths
    fake_paths = [os.path.join('fake_datasets', tool, f) for f in fake_datasets]

    # Iterate through each dataset and evaluate
    for fake_dataset in fake_datasets:
        fake_dataset_path = os.path.join('fake_datasets', tool, fake_dataset)

        # Classification or Regression Evaluation
        print(f'Evaluating ML Utility for {fake_dataset}')
        if task_type == 'classification':
            classifiers.evaluate_all_datasets(target_column, dataset, tool, performance_dir)
        elif task_type == 'regression':
            regressors.evaluate_all_datasets(target_column, dataset, tool, performance_dir)
        else:
            raise ValueError("Unknown task type. Please specify either 'classification' or 'regression'.")

        # Evaluate similarity
        print(f'Evaluating Similarity for {fake_dataset}')
        real_path = os.path.join('data', f'{dataset}.csv')
        similarity_main(real_path, fake_paths, dataset, tool, performance_dir, categorical_columns, continuous_columns)

    # Evaluate statistics (now handles multiple datasets)
    print(f'Evaluating Statistics for all datasets of {dataset}')
    statistics_main(dataset, tool, performance_dir, categorical_columns, continuous_columns)

    # Evaluate detection metrics (now handles multiple datasets)
    print(f'Evaluating Detection for all datasets of {dataset}')
    detection_main(real_path, dataset, tool, performance_dir, subsample_size=subsample_size)

    # Evaluate privacy metrics (now handles multiple datasets)
    print(f'Evaluating Privacy for all datasets of {dataset}')
    privacy_main(real_path, dataset, tool, performance_dir)

    # Evaluate correlations (added evaluation for correlations)
    print(f'Evaluating Correlations for all datasets of {dataset}')
    correlations_main(dataset, tool, performance_dir, categorical_columns, continuous_columns)

def process_multiple_tools(json_path):
    tools_dir = 'tools'
    data_dir = 'data'
    output_dir = 'fake_datasets'
    performance_dir = 'performance'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(performance_dir, exist_ok=True)

    # Read the experiment JSON file
    with open(json_path, 'r') as file:
        experiment_data = json.load(file)

    # Ensure the JSON is a list of dictionaries
    if not isinstance(experiment_data, list):
        raise ValueError("The JSON file should be a list of dictionaries.")

    for experiment in experiment_data:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment should be a dictionary.")
        
        tool_choice = experiment.get('toolname')
        dataset_choice = experiment.get('dataset')
        task_type = experiment.get('problem_type')
        target_column = experiment.get('target')
        subsample_size = experiment.get('subsample_size', 1000)  # Default to 1000

        columns = experiment.get('columns', {})
        categorical_columns = columns.get('categorical', [])
        continuous_columns = columns.get('continuous', [])

        if not all([tool_choice, dataset_choice, task_type, target_column]):
            raise ValueError("Each experiment must include 'toolname', 'dataset', 'problem_type', 'target'.")
        
        try:
            # Create and activate conda environment for the tool
            tool_path = os.path.join(tools_dir, f'{tool_choice}-main')
            create_and_activate_conda_env(tool_choice, tool_path)

            # Start monitoring
            print("Starting monitoring...")
            monitor_process = start_monitoring()

            # Load the real dataset
            dataset_path = os.path.join(data_dir, f'{dataset_choice}.csv')
            dataset = load_dataset(dataset_path)

            # Create the tool-specific output directory
            tool_output_dir = os.path.join(output_dir, tool_choice)
            os.makedirs(tool_output_dir, exist_ok=True)
            output_path = os.path.join(tool_output_dir)

            # Run the tool (this will generate multiple fake datasets)
            output_path = os.path.join(output_dir, tool_choice)
            run_tool(tool_path, dataset_choice, output_path, json_path)

            # Stop monitoring
            print("Stopping monitoring...")
            stop_monitoring(monitor_process)

            # Create a tool-specific performance directory
            performance_tool_dir = os.path.join(performance_dir, tool_choice)
            os.makedirs(performance_tool_dir, exist_ok=True)

            # Read and save CPU and memory performance metrics
            cpu_mem_metrics = read_cpu_mem_performance_metrics()
            cpu_mem_metrics_df = pd.DataFrame([cpu_mem_metrics])
            cpu_mem_performance_path = os.path.join(performance_tool_dir, f'{tool_choice}_{dataset_choice}_performance.csv')
            cpu_mem_metrics_df.to_csv(cpu_mem_performance_path, index=False)

            # Read and save GPU performance metrics
            gpu_metrics_df = read_gpu_performance_metrics()
            gpu_performance_path = os.path.join(performance_tool_dir, f'{tool_choice}_{dataset_choice}_gpu_performance.csv')
            gpu_metrics_df.to_csv(gpu_performance_path, index=False)

            # Skip evaluation for 'tabddpm' or 'smote'
            if tool_choice not in ["tabddpm", "smote", "tabsyn"]:
                # Evaluate synthetic datasets
                print(f'Evaluating {tool_choice}_{dataset_choice}')
                evaluate_synthetic_data(tool_choice, dataset_choice, performance_tool_dir, task_type, target_column, categorical_columns, continuous_columns, subsample_size)
            else:
                print(f"Skipping evaluation for {tool_choice} as it's either 'tabddpm', 'tabsyn' or 'smote'.")

            print(f"Benchmarking for {tool_choice} on {dataset_choice} completed.")

        except Exception as e:
            print(f"An error occurred for {tool_choice} on {dataset_choice}: {e}")


def main():
    print("Please provide the path to the experiment JSON file:")
    json_path = input().strip()
    if os.path.exists(json_path):
        process_multiple_tools(json_path)
    else:
        print("The specified JSON file does not exist.")

if __name__ == "__main__":
    main()
