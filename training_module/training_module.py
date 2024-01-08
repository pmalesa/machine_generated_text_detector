import os
import numpy as np
import pandas as pd
from datetime import datetime
from io import StringIO
from pandas import DataFrame
import yaml

from models.generated_text_detection_model.generated_text_detection_model import GeneratedTextDetectionModel

class TrainingModule:
    def __init__(self):
        self.__data: DataFrame
        self.__config = self.__load_config("parameters.yaml")
    
    def train_gtd(self):
        # Read gtd model parameters
        self.__config = self.__config["gtd"]

        print("[INFO] Generated text detection model training started.")
        print("[INFO] Loading training data...")
        self.__data = self.__load_data(self.__config["train_data_path"])

        if not self.__data.empty:
            print("[INFO] Training data loaded sucessfully.")
        else:
            print("[ERROR] Training data could not be loaded.")
            return

        gtd_model = GeneratedTextDetectionModel()

        # Model training
        start = datetime.now()
        gtd_model.train(
            self.__data["text"],
            self.__data["label"],
            self.__config["learning_rate"],
            self.__config["epochs"],
            self.__config["chunk_size"],
            self.__config["chunk_overlap"]
        )
        finish = datetime.now()
        duration = finish - start
        training_time_in_s = duration.total_seconds()

        # Save model
        directory = "output/gtd"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok = True)
        
        model_name = f"model_weights-epochs({self.__config['epochs']})_chunk_size({self.__config['chunk_size']})_chunk_overlap({self.__config['chunk_overlap']}).h5"
        model_save_path = os.path.join(directory, model_name)
        gtd_model.save_weights(model_save_path)

        print("[INFO] Training of GTD model finished!")
        print(f"[INFO] Training time [min]: {training_time_in_s / 60}")

    def train_lmd(self):
        print("[INFO] Generated text detection model training started.")
        print("[INFO] Loading training data...")
        self.__data = self.__load_data("data/subtask_B/subtaskB_train.jsonl")

        if not self.__data.empty:
            print("[INFO] Training data loaded sucessfully.")
        else:
            print("[ERROR] Training data could not be loaded.")
            return

        pass

    def train_tsd(self):
        print("[INFO] Generated text detection model training started.")
        print("[INFO] Loading training data...")
        self.__data = self.__load_data("data/subtask_C/subtaskC_train.jsonl")

        if not self.__data.empty:
            print("[INFO] Training data loaded sucessfully.")
        else:
            print("[ERROR] Training data could not be loaded.")
            return

        pass

    def __load_data(self, filename: str) -> DataFrame:
        with open(filename, "r") as file:
            jsonl_string = file.read()
            jsonl_io = StringIO(jsonl_string)
            return pd.read_json(jsonl_io, lines = True)
        
    def __load_config(self, config_file: str):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)