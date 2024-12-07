import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import norm

# Function to subsample real and fake data to the same size
def fixed_subsample(real_data, fake_data, subsample_size=1000, seed=42):
    np.random.seed(seed)
    
    min_size = min(len(real_data), len(fake_data), subsample_size)
    
    if min_size < 500:
        real_subsample = real_data
        fake_subsample = fake_data
    else:
        real_indices = np.random.choice(len(real_data), min_size, replace=False)
        fake_indices = np.random.choice(len(fake_data), min_size, replace=False)
        
        real_subsample = real_data[real_indices]
        fake_subsample = fake_data[fake_indices]
    
    return real_subsample, fake_subsample

# Function to compute Euclidean distances
def compute_pairwise_distances(x, y=None):
    if y is None:
        y = x
    return euclidean_distances(x, y)

# Compute PRDC metrics
def compute_prdc(real_features, fake_features, k=5, subsample_size=1000, seed=42):
    real_features_subsample, fake_features_subsample = fixed_subsample(real_features, fake_features, subsample_size=subsample_size, seed=seed)

    distances_real = compute_pairwise_distances(real_features_subsample)
    distances_fake_to_real = compute_pairwise_distances(fake_features_subsample, real_features_subsample)

    precision = compute_precision(distances_real, distances_fake_to_real, k)
    recall = compute_recall(distances_real, distances_fake_to_real, k)
    density = compute_density(distances_real, distances_fake_to_real, k)
    coverage = compute_coverage(distances_real, distances_fake_to_real, k)

    return precision, recall, density, coverage

# Helper functions to calculate PRDC metrics
def compute_precision(real_distances, fake_to_real_distances, k):
    nearest_k_distances_real = np.partition(real_distances, k, axis=1)[:, :k]
    thresholds = np.max(nearest_k_distances_real, axis=1)
    thresholds = thresholds[None, :]
    precision = np.mean(np.any(fake_to_real_distances < thresholds, axis=1))
    return precision

def compute_recall(real_distances, fake_to_real_distances, k):
    nearest_k_distances_real = np.partition(real_distances, k, axis=1)[:, :k]
    thresholds = np.max(nearest_k_distances_real, axis=1)
    thresholds = thresholds[None, :]
    recall = np.mean(np.any(fake_to_real_distances < thresholds, axis=1))
    return recall

def compute_density(real_distances, fake_to_real_distances, k):
    nearest_k_distances_real = np.partition(real_distances, k, axis=1)[:, :k]
    thresholds = np.max(nearest_k_distances_real, axis=1)
    density = np.mean(np.sum(fake_to_real_distances < thresholds[None, :], axis=1) / k)
    return density

def compute_coverage(real_distances, fake_to_real_distances, k):
    nearest_k_distances_real = np.partition(real_distances, k, axis=1)[:, :k]
    thresholds = np.max(nearest_k_distances_real, axis=1)
    coverage = np.mean(np.any(fake_to_real_distances < thresholds[None, :], axis=1))
    return coverage

# Compute Alpha Precision Metric
def compute_alpha_precision(real_features, fake_features, alpha=0.05, subsample_size=1000, seed=42):
    real_features_subsample, fake_features_subsample = fixed_subsample(real_features, fake_features, subsample_size=subsample_size, seed=seed)
    distances_fake_to_real = compute_pairwise_distances(fake_features_subsample, real_features_subsample)
    sorted_distances = np.sort(distances_fake_to_real, axis=0)
    threshold = sorted_distances[int(alpha * len(sorted_distances))]
    alpha_precision = np.mean(distances_fake_to_real < threshold)
    return alpha_precision

# Detection using various models
def detection_with_model(real_features, fake_features, model, subsample_size=1000, seed=42):
    real_features_subsample, fake_features_subsample = fixed_subsample(real_features, fake_features, subsample_size=subsample_size, seed=seed)
    X = np.vstack((real_features_subsample, fake_features_subsample))
    y = np.hstack((np.ones(len(real_features_subsample)), np.zeros(len(fake_features_subsample))))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function to evaluate detection metrics for all datasets
def evaluate_all_datasets(real_path, dataset_name, tool_name, performance_dir, subsample_size=1000):
    fake_datasets_dir = os.path.join('fake_datasets', tool_name)
    fake_paths = [os.path.join(fake_datasets_dir, f"{tool_name}_{dataset_name}_{i}.csv") for i in range(1, 6)]
    all_results = []
    detailed_jsons = []

    for i, fake_path in enumerate(fake_paths, 1):
        real_data = pd.read_csv(real_path)
        fake_data = pd.read_csv(fake_path)

        real_features = real_data.values
        fake_features = fake_data.values

        # PRDC metrics
        precision, recall, density, coverage = compute_prdc(real_features, fake_features, subsample_size=subsample_size)

        # Alpha Precision
        alpha_precision = compute_alpha_precision(real_features, fake_features, subsample_size=subsample_size)

        # Detection metrics using different models
        logistic_regression_accuracy = detection_with_model(real_features, fake_features, LogisticRegression(), subsample_size=subsample_size)
        nn_accuracy = detection_with_model(real_features, fake_features, MLPClassifier(random_state=42), subsample_size=subsample_size)
        xgb_accuracy = detection_with_model(real_features, fake_features, XGBClassifier(random_state=42), subsample_size=subsample_size)
        gmm_accuracy = detection_with_model(real_features, fake_features, GaussianMixture(n_components=2, random_state=42), subsample_size=subsample_size)

        # Save detailed metrics for each dataset
        detailed_metrics = {
            'Dataset': f"{dataset_name}_{i}",
            'TDS Model': tool_name,
            'PRDC': {
                'Precision': precision,
                'Recall': recall,
                'Density': density,
                'Coverage': coverage
            },
            'Alpha Precision': alpha_precision,
            'Detection Accuracy': {
                'Logistic Regression': logistic_regression_accuracy,
                'Neural Network': nn_accuracy,
                'XGBoost': xgb_accuracy,
                'GMM': gmm_accuracy
            }
        }
        detailed_jsons.append(detailed_metrics)

    # Save all detailed metrics to a single JSON file
    detailed_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_detection_evaluation_detailed.json")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_jsons, f, indent=4)

    # Collect results for overall averages
    avg_metrics = {
        'Dataset': f"{dataset_name}_average",
        'TDS Model': tool_name,
        'PRDC': {
            'Precision': np.mean([res['PRDC']['Precision'] for res in detailed_jsons]),
            'Recall': np.mean([res['PRDC']['Recall'] for res in detailed_jsons]),
            'Density': np.mean([res['PRDC']['Density'] for res in detailed_jsons]),
            'Coverage': np.mean([res['PRDC']['Coverage'] for res in detailed_jsons])
        },
        'Alpha Precision': np.mean([res['Alpha Precision'] for res in detailed_jsons]),
        'Detection Accuracy': {
            'Logistic Regression': np.mean([res['Detection Accuracy']['Logistic Regression'] for res in detailed_jsons]),
            'Neural Network': np.mean([res['Detection Accuracy']['Neural Network'] for res in detailed_jsons]),
            'XGBoost': np.mean([res['Detection Accuracy']['XGBoost'] for res in detailed_jsons]),
            'GMM': np.mean([res['Detection Accuracy']['GMM'] for res in detailed_jsons])
        }
    }

    # Save the average metrics to a JSON file
    avg_output_filename = os.path.join(performance_dir, f"{tool_name}_{dataset_name}_detection_averages.json")
    with open(avg_output_filename, 'w') as f:
        json.dump(avg_metrics, f, indent=4)

    #print(f"Detection evaluation averages saved to '{avg_output_filename}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate detection metrics for synthetic datasets.")
    parser.add_argument("real_path", type=str, help="Path to the real data CSV file")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("tool_name", type=str, help="Name of the TDS model")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    parser.add_argument("--subsample_size", type=int, default=1000, help="Subsample size for both real and fake data")
    args = parser.parse_args()

    evaluate_all_datasets(args.real_path, args.dataset_name, args.tool_name, args.performance_dir, args.subsample_size)
