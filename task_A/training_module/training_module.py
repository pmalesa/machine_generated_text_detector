import numpy as np
import pandas as pd

from io import StringIO
from pandas import DataFrame, Series

from text_processing.text_processor import TextProcessor as tp

class TrainingModule:
    def __init__(self):
        self.__filename: str = "data/subtaskA_train_monolingual.jsonl"

        print("[INFO] Loading training data...")
        self.__df = self.__load_data()
        print("[INFO] Training data loaded.")
    
    def run(self):
        print("[INFO] Training started.")
        texts_vec = self.__df["text"]
        labels_vec = self.__df["label"]

        print(f"Example 1: {texts_vec[0]}\n Label: {labels_vec[0]}")
        print(tp.tokenize_text(texts_vec[0]))




    def __load_data(self) -> DataFrame:
        with open(self.__filename, "r") as file:
            jsonl_string = file.read()
            jsonl_io = StringIO(jsonl_string)
            return pd.read_json(jsonl_io, lines = True)
        
    def __preprocess_text_data(self, texts_vec: Series) -> np.array:
        pass