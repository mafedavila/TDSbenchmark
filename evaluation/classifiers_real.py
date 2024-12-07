import os
import pandas as pd
import numpy as np
import json
from math import nan
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Function to evaluate each classifier
def evaluate_classifier(name, model, X_train, X_test, y_train, y_test, results):
    # print(f"Training and evaluating classifier: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        acc = accuracy_score(y_test, y_pred)
    except ValueError as e:
        # print(f"Error calculating accuracy for {name}: {e}")
        acc = nan

    try:
        auc = roc_auc_score(y_test, y_pred)
    except ValueError as e:
        # print(f"Error calculating AUC for {name}: {e}")
        auc = nan

    try:
        f1score_micro = f1_score(y_test, y_pred, average="micro")
    except ValueError as e:
        # print(f"Error calculating F1 Score Micro for {name}: {e}")
        f1score_micro = nan

    try:
        f1score_macro = f1_score(y_test, y_pred, average="macro")
    except ValueError as e:
        # print(f"Error calculating F1 Score Macro for {name}: {e}")
        f1score_macro = nan

    try:
        f1score_we = f1_score(y_test, y_pred, average="weighted")
    except ValueError as e:
        # print(f"Error calculating F1 Score Weighted for {name}: {e}")
        f1score_we = nan

    results.append({
        "Classifier": name,
        "Accuracy": acc,
        "AUC": auc,
        "F1 Score Micro": f1score_micro,
        "F1 Score Macro": f1score_macro,
        "F1 Score Weighted": f1score_we
    })
    # print(f"Results for {name}: {results[-1]}")

# Function to evaluate classifiers on a single dataset
def classifiers_evaluation(path, predicted, dataset, ds_model):
    # print(f"Loading dataset: {path}")
    my_df = pd.read_csv(path)
    my_df = my_df.head(10000)
    # print(f"Dataset loaded: {len(my_df)} rows")

    y = my_df[predicted]
    drop_elements = [predicted]
    X = my_df.drop(drop_elements, axis=1)

    # print(f"Splitting data into train and test sets for {dataset}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(f"Training set: {len(X_train)} rows, Test set: {len(X_test)} rows")

    # List of classifiers to evaluate
    classifiers = [
        ("Perceptron", Perceptron(random_state=42)),
        ("MLP", MLPClassifier(random_state=42)),
        ("Gaussian NB", GaussianNB()),
        ("Linear SVM", svm.SVC(random_state=42, gamma=0.22)),
        ("Radical SVM", svm.SVC(random_state=42, kernel='rbf', C=1, gamma=0.22)),
        ("Log Reg", LogisticRegression(random_state=42)),
        ("RF", RandomForestClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier()),
        ("DT", DecisionTreeClassifier(random_state=42))
    ]

    results = []
    for name, model in classifiers:
        evaluate_classifier(name, model, X_train, X_test, y_train, y_test, results)

    return results

# Function to evaluate all fake datasets and calculate averages
def evaluate_all_datasets(predicted, dataset_name, ds_model, performance_dir):
    # Correct directory to search for datasets
    fake_datasets_dir = os.path.join('fake_datasets', ds_model)
    
    # print(f"Searching for datasets matching pattern: {ds_model}_{dataset_name}_*.csv in {fake_datasets_dir}")
    
    # Find all the files that match the pattern toolname_dataset_n.csv (where n is 1, 2, 3...)
    fake_files = sorted([f for f in os.listdir(fake_datasets_dir) 
                         if f.startswith(f"{ds_model}_{dataset_name}_") and f.endswith('.csv')])

    fake_paths = [os.path.join(fake_datasets_dir, f) for f in fake_files]

    if not fake_paths:
        # print(f"No datasets found for {ds_model} and {dataset_name} in {fake_datasets_dir}")
        return

    # print(f"Found {len(fake_paths)} datasets: {fake_files}")

    all_results = []
    detailed_results = {}  # Dictionary to store all detailed results

    # Evaluate each fake dataset
    for i, fake_path in enumerate(fake_paths, 1):
        # print(f"Evaluating dataset {fake_path}...")
        results = classifiers_evaluation(fake_path, predicted, f"{dataset_name}_{i}", ds_model)
        detailed_results[f"Fake dataset {i}"] = results  # Add results under the "Fake dataset" key

        # Collect results to compute averages
        all_results.append(results)

    # Save all detailed results to a single JSON file
    detailed_output_filename = os.path.join(performance_dir, f"{ds_model}_{dataset_name}_classification_evaluation_detailed.json")
    # print(f"Saving all detailed results to {detailed_output_filename}")
    with open(detailed_output_filename, 'w') as f:
        json.dump(detailed_results, f, indent=4)

    # Combine all results into a DataFrame for easy averaging
    combined_results = []
    for result_set in all_results:
        for result in result_set:
            combined_results.append(result)

    combined_df = pd.DataFrame(combined_results)

    # Calculate the average for each classifier across the datasets
    # print(f"Calculating average metrics across {len(fake_paths)} datasets")
    average_results = combined_df.groupby("Classifier").mean().reset_index()

    # Save the average metrics to a CSV file
    avg_output_filename = os.path.join(performance_dir, f"{ds_model}_{dataset_name}_classification_evaluation_average.csv")
    # print(f"Saving average metrics to {avg_output_filename}")
    average_results.to_csv(avg_output_filename, index=False)

    # print(f"Saved average classifier evaluation results to {avg_output_filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate classifiers on datasets")
    parser.add_argument("real_dataset_path", type=str, help="Path to the real dataset CSV")
    parser.add_argument("predicted", type=str, help="The target column to predict")
    parser.add_argument("performance_dir", type=str, help="Directory to save performance metrics")
    args = parser.parse_args()

    # Extract the dataset name from the real dataset path
    dataset_name = os.path.splitext(os.path.basename(args.real_dataset_path))[0]  # Get file name without extension

    # Evaluate the real dataset
    real_results = classifiers_evaluation(args.real_dataset_path, args.predicted, dataset_name, "real")

    # Save detailed results for the real dataset
    real_detailed_output = os.path.join(
        args.performance_dir, f"real_{dataset_name}_classification_evaluation_detailed.json"
    )
    with open(real_detailed_output, 'w') as f:
        json.dump({"Real Dataset": real_results}, f, indent=4)

    # Combine results into a DataFrame
    real_df = pd.DataFrame(real_results)

    # Save average metrics for the real dataset
    real_avg_output = os.path.join(
        args.performance_dir, f"real_{dataset_name}_classification_evaluation_average.csv"
    )
    real_df.to_csv(real_avg_output, index=False)

    print(f"Real dataset evaluation results saved to:")
    print(f" - Detailed JSON: {real_detailed_output}")
    print(f" - Average CSV: {real_avg_output}")