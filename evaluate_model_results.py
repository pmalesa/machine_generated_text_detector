import os
import argparse
import yaml
import json
import pandas as pd
from io import StringIO
from pandas import DataFrame

def load_data(filename: str) -> DataFrame:
    with open(filename, "r") as file:
        jsonl_string = file.read()
        jsonl_io = StringIO(jsonl_string)
        return pd.read_json(jsonl_io, lines = True)

def load_config(config_file: str):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
def calculate_metrics_gtd(predicted_labels: list[int], true_labels: list[int]):
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    tp, tn, fp, fn = 0, 0, 0, 0

    for i, (predicted_label, true_label) in enumerate(zip(predicted_labels, true_labels)):
        if predicted_label == 1 and true_label == 1:
            tp += 1
        elif predicted_label == 1 and true_label == 0:
            fp += 1
        elif predicted_label == 0 and true_label == 0:
            tn += 1
        else:
            fn += 1

    if tp + tn + fp + fn> 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy = -1.0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = -1.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = -1.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = -1.0

    return accuracy, precision, recall, f1

def calculate_metrics_lmd(predicted_labels: list[int], true_labels: list[int]):
    correct = 0
    for i, (predicted_label, true_label) in enumerate(zip(predicted_labels, true_labels)):
        if predicted_label == 1 and true_label == 1:
            correct += 1

    return correct / len(true_labels)
    
def run(model_name: str):
    print(f"Evaluation metrics of model: {model_name}")
    
    config = load_config("parameters.yaml")
    true_labels = load_data(config[model_name]["test_data_path"])["label"]
    predicted_labels = load_data(config[model_name]["predictions_path"])["label"]

    if (model_name == "gtd"):
        accuracy, precission, recall, f1 = calculate_metrics_gtd(predicted_labels, true_labels)
        print(f"\nAccuracy: {accuracy}")
        print(f"Precision: {precission}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")
    elif (model_name == "lmd"):
        accuracy = calculate_metrics_lmd(predicted_labels, true_labels)
        print(f"\nAccuracy: {accuracy}")
    else:
        config = config["tsd"]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the script by choosing the specific model to test.")
    parser.add_argument("model", choices = ["gtd", "lmd", "tsd"], help = "Model to run the script with. Options are \"gtd\", \"lmd\", or \"tsd\")")
    args = parser.parse_args()
    run(args.model)
