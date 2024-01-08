import os
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

from io import StringIO
from pandas import DataFrame

from models.generated_text_detection_model.generated_text_detection_model import GeneratedTextDetectionModel

class TestingModule:
    def __init__(self):
        self.__data: DataFrame
        self.__config = self.__load_config("parameters.yaml")   

    def test_gtd(self):
        # Read gtd model parameters
        self.__config = self.__config["gtd"]

        print("[INFO] Generated text detection model testing started.")
        print("[INFO] Loading testing data...")
        self.__data = self.__load_data(self.__config["test_data_path"])

        if not self.__data.empty:
            print("[INFO] Testing data loaded sucessfully.")
        else:
            print("[ERROR] Testing data could not be loaded.")
            return
        
        model_path = self.__config["weights_path"] + "/" + self.__config["weights_filename"]
        if not os.path.exists(model_path):
            print("[ERROR] Model file does not exist.")
            return

        # Model testing
        gtd_model = GeneratedTextDetectionModel()
        gtd_model.load(model_path, self.__config["chunk_size"], self.__config["chunk_overlap"])
        gtd_model.test(self.__data["text"], self.__data["label"])
        print("[INFO] Testing of GTD model finished!")

    def __load_data(self, filename: str) -> DataFrame:
        with open(filename, "r") as file:
            jsonl_string = file.read()
            jsonl_io = StringIO(jsonl_string)
            return pd.read_json(jsonl_io, lines = True)
        
    def __load_config(self, config_file: str):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)