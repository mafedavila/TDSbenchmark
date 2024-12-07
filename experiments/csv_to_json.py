import csv
import json
import sys

def csv_to_json(csv_file_path, json_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Create a list to hold the JSON data
        json_data = []
        
        for row in csv_reader:
            entry = {
                "toolname": row["toolname"].strip(),
                "dataset": row["dataset"].strip(),
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
            json_data.append(entry)
        
        with open(json_file_path, mode='w') as json_file:
            json.dump(json_data, json_file, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_json.py <input_csv_file> <output_json_file>")
    else:
        csv_file_path = sys.argv[1]
        json_file_path = sys.argv[2]
        csv_to_json(csv_file_path, json_file_path)
