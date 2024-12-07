import csv
import json
import sys
import os

def csv_to_json_per_tool_and_dataset(csv_file_path, output_dir):
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Print headers to diagnose the issue
        headers = csv_reader.fieldnames
        print("Headers found in the CSV file:", headers)
        
        # Create dictionaries to hold JSON data for each toolname and dataset
        tools_data = {}
        datasets_data = {}
        
        for row in csv_reader:
            toolname = row["toolname"].strip()
            dataset = row["dataset"].strip()
            
            if toolname not in tools_data:
                tools_data[toolname] = []
            
            if dataset not in datasets_data:
                datasets_data[dataset] = []
            
            entry = {
                "toolname": toolname,
                "dataset": dataset,
                "problem_type": row["problem_type"].strip(),
                "target": row["target"].strip(),
                "columns": {
                    "categorical": row["categorical columns"].split(';') if row["categorical columns"].strip() else [],
                    "continuous": row["continuous columns"].split(';') if row["continuous columns"].strip() else [],
                    "integer": row["integer columns"].split(';') if row["integer columns"].strip() else [],
                    "mixed": {k: [float(v.strip('[]'))] for k, v in (item.split(':') for item in row["mixed columns"].split(';'))} if row["mixed columns"].strip() else {},
                    "log": row["log columns"].split(';') if row["log columns"].strip() else [],
                    "general": row["general columns"].split(';') if row["general columns"].strip() else [],
                    "non_categorical": row["non_categorical columns"].split(';') if row["non_categorical columns"].strip() else []
                }
            }
            tools_data[toolname].append(entry)
            datasets_data[dataset].append(entry)
        
        # Ensure the output directories exist
        tool_output_dir = os.path.join(output_dir, 'per_tool')
        dataset_output_dir = os.path.join(output_dir, 'per_dataset')
        os.makedirs(tool_output_dir, exist_ok=True)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Write each toolname's data to its own JSON file
        for toolname, data in tools_data.items():
            json_file_path = os.path.join(tool_output_dir, f"{toolname}.json")
            with open(json_file_path, mode='w') as json_file:
                json.dump(data, json_file, indent=2)
        
        # Write each dataset's data to its own JSON file
        for dataset, data in datasets_data.items():
            json_file_path = os.path.join(dataset_output_dir, f"{dataset}.json")
            with open(json_file_path, mode='w') as json_file:
                json.dump(data, json_file, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_json_per_tool_and_dataset.py <input_csv_file> <output_directory>")
    else:
        csv_file_path = sys.argv[1]
        output_dir = sys.argv[2]
        csv_to_json_per_tool_and_dataset(csv_file_path, output_dir)
