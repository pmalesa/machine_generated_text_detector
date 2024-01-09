import os
import argparse
import yaml
import json

from models.generated_text_detection_model.generated_text_detection_model import GeneratedTextDetectionModel
from models.language_model_detection_model.language_model_detection_model import LanguageModelDetectionModel

def load_config(config_file: str):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
def read_example(file_path: str, text_id: int):
    print(file_path)
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("id") == text_id:
                return data
    raise ValueError(f"ID {text_id} not found in the file.")
    
def run(model_name: str, text_id_or_text: str):
    print(f"Chosen model: {model_name}")
    
    config = load_config("parameters.yaml")
    model = None
    prediction = None
    true_label = None
    text_id = None
    text = None

    try:
        text_id = int(text_id_or_text)
    except ValueError:
        text = text_id_or_text

    if (model_name == "gtd"):
        config = config["gtd"]
        model = GeneratedTextDetectionModel()
    elif (model_name == "lmd"):
        config = config["lmd"]
        model = LanguageModelDetectionModel()
    else:
        config = config["tsd"]

    if model == None:
        print(f"[ERROR] There was a problem creating a {model_name} model.")
        return
    
    model_path = config["weights_path"] + "/" + config["weights_filename"]
    if not os.path.exists(model_path):
        print("[ERROR] Model file does not exist.")
        return
    
    model.load(
        model_path,
        config["chunk_size"],
        config["chunk_overlap"]
    )
    try:
        if text_id is not None:
            example = read_example(config["test_data_path"], text_id)
            true_label = example["label"]        
            prediction = model.predict_single(example["text"])
            print("\n--------------------")
            print(f"Text ID: {text_id}")
            print(f"Prediction: {prediction}")
            print(f"True label: {true_label}")
        else:
            prediction = model.predict_single(text)
            print("\n--------------------")
            print(f"Prediction: {prediction}")
        if prediction == None:
            print("[ERROR] There was a problem with conducting prediction.")
            return
    except ValueError as e:
        print(e)
        return
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the script by choosing the specific model to test.")
    parser.add_argument("model", choices = ["gtd", "lmd", "tsd"], help = "Model to run the script with. Options are \"gtd\", \"lmd\", or \"tsd\")")
    parser.add_argument("text_id_or_text", help="Id (integer) or name (string) of an example from the test set.")
    args = parser.parse_args()
    run(args.model, args.text_id_or_text)
