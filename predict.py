import os
import argparse
import yaml
import json

from models.generated_text_detection_model.generated_text_detection_model import GeneratedTextDetectionModel

def load_config(config_file: str):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
def read_example(file_path: str, text_id: int):
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("id") == text_id:
                return data
    raise ValueError(f"ID {text_id} not found in the file.")
    
def run(model_name: str, text_id: int):
    print(f"Chosen model: {model_name}")
    
    model = GeneratedTextDetectionModel()
    config = load_config("parameters.yaml")

    prediction = None
    true_label = None

    if (model_name == "gtd"):
        config = config["gtd"]
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
            example = read_example(config["test_data_path"], text_id)
        except ValueError as e:
            print(e)

        true_label = example["label"]        
        prediction = model.predict_single(example["text"])

    elif (model_name == "lmd"):
        config = config["lmd"]
        return
    else:
        config = config["tsd"]
        return
    
    if prediction == None or true_label == None:
        print("[ERROR] There was a problem with conducting prediction.")
        return
    
    print("\n--------------------")
    print(f"Text ID: {text_id}")
    print(f"Prediction: {prediction}")
    print(f"True label: {true_label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the script by choosing the specific model to test.")
    parser.add_argument("model", choices = ["gtd", "lmd", "tsd"], help = "Model to run the script with. Options are \"gtd\", \"lmd\", or \"tsd\")")
    parser.add_argument("text_id", type = int, help = "Id of an example from the test set.")
    args = parser.parse_args()
    run(args.model, args.text_id)
