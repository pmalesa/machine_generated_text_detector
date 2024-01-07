import os
import numpy as np
import pandas as pd
from datetime import datetime

from io import StringIO
from pandas import DataFrame

from text_processing.text_processor import TextProcessor as tp
from models.generated_text_detection_model.generated_text_detection_model import GeneratedTextDetectionModel

class TrainingModule:
    def __init__(self):
        self.__data: DataFrame
    
    def train_gtd(self):
        print("[INFO] Generated text detection model training started.")
        print("[INFO] Loading training data...")
        self.__data = self.__load_data("data/subtask_A/subtaskA_train_monolingual.jsonl")

        if not self.__data.empty:
            print("[INFO] Training data loaded sucessfully.")
        else:
            print("[ERROR] Training data could not be loaded.")
            return

        # Model training
        gtd_model = GeneratedTextDetectionModel()
        learning_rate = 0.01
        epochs = 1
        chunk_size = 128
        chunk_overlap = 32

        start = datetime.now()
        gtd_model.train(self.__data["text"], self.__data["label"], learning_rate, epochs, chunk_size, chunk_overlap)
        finish = datetime.now()
        duration = finish - start
        training_time_in_s = duration.total_seconds()

        # Save model
        directory = "output/gtd"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok = True)
        
        model_name = f"model_weights-epochs({epochs})_chunk_size({chunk_size})_chunk_overlap({chunk_overlap}).h5"
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
        