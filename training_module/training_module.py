import numpy as np
import pandas as pd

from io import StringIO
from pandas import DataFrame, Series

from text_processing.text_processor import TextProcessor as tp

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

        texts_vec = self.__data["text"]
        labels_vec = self.__data["label"]

        print(f"Number of training examples: {self.__data.shape[0]}")

        print(f"Example 1: {texts_vec[0]}\n Label: {labels_vec[0]}")
        print(tp.tokenize_text(texts_vec[0]))



    # MAX WORD COUNT IN A TEXT IS 38959, SO LET'S SET CONSTRAINT TO 50k


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
        
    def __preprocess_text_data(self, texts_vec: Series) -> np.array:
        pass