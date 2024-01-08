import os
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import json

from text_processing.text_processor import TextProcessor as tp


tf.config.run_functions_eagerly(True)

class GeneratedTextDetectionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.__tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.__deberta_model = TFAutoModel.from_pretrained("microsoft/deberta-base")
        self.__deberta_model.trainable = False
        self.__intermediate_dense_layer = tf.keras.layers.Dense(768, activation = "relu")
        self.__output_dense_layer = tf.keras.layers.Dense(1, activation = "sigmoid")

        self.__learning_rate = 0.001
        self.__epochs = 1
        self.__chunk_size = 512
        self.__chunk_overlap = 64
        self.__max_length = 512

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training = False):
        outputs = self.__deberta_model(inputs)
        cls_output = outputs[0][:, 0, :]
        intermediate_output = self.__intermediate_dense_layer(cls_output)
        return self.__output_dense_layer(intermediate_output)

    def init(self, learning_rate: float = 0.001, epochs: int = 1, chunk_size: int = 512, chunk_overlap: int = 64):
        self.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
            loss = tf.keras.losses.BinaryCrossentropy(),
            metrics = [tf.keras.metrics.BinaryAccuracy()]
        )

        self.__learning_rate = learning_rate
        self.__epochs = epochs
        self.__chunk_size = chunk_size
        self.__chunk_overlap = chunk_overlap
        self.__max_length = 2 * chunk_size
        if self.__max_length > 512:
            self.__max_length = 512

        dummy_data = self.__tokenizer(
            ["This is a dummy input for initialization"],
            max_length = self.__max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "tf"
        )
        self(dummy_data)

    def train(self, texts: list[str], labels: list[int], learning_rate = 0.001, epochs = 10, chunk_size = 512, chunk_overlap = 64):
        if len(texts) != len(labels):
            print("[ERROR] Number of examples is not equal to the number of labels!")
            return
        
        self.init(learning_rate, epochs, chunk_size, chunk_overlap)

        # Training the model
        for i, (text, label) in enumerate(zip(texts, labels)):
            print(f"[Example {i + 1}]")
            self.__train_single(text, label)

    def test(self, texts: list[str], true_labels: list[int]):
        predictions = []
        for i, text in enumerate(texts):
            print(f"[Example {i + 1}]")
            prediction = self.predict_single(text)
            if prediction != -1:
                predictions.append(prediction)

        # Save predictions
        file_path = "gtd_predictions.jsonl"
        directory = "output/gtd"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok = True)
        path = os.path.join(directory, file_path)
        with open(path, "w") as file:
            for i, prediction in enumerate(predictions):
                line = {"id": i, "label": prediction}
                json_line = json.dumps(line)
                file.write(json_line + "\n")

    def load(self, path: str, chunk_size = 512, chunk_overlap = 64):
        self.init(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        self.load_weights(path)

    def __train_single(self, text: str, label: int):
        text_chunks = tp.preprocess_text(
            text,
            chunk_size = self.__chunk_size,
            chunk_overlap = self.__chunk_overlap
        )

        if len(text_chunks) == 0:
            return

        chunks_features = self.__tokenizer(
            text_chunks,
            padding = "max_length",
            truncation = True,
            max_length = self.__max_length,
            return_tensors = "tf"
        )
        for i in range(len(chunks_features["input_ids"])):
            chunk_features = {
                "input_ids": tf.expand_dims(chunks_features["input_ids"][i], 0),
                "attention_mask": tf.expand_dims(chunks_features["attention_mask"][i], 0),
                "token_type_ids": tf.expand_dims(chunks_features["token_type_ids"][i], 0)
            }
            chunk_label = tf.constant([label])
            chunk_dataset = tf.data.Dataset.from_tensor_slices((dict(chunk_features), chunk_label))
            chunk_dataset = chunk_dataset.batch(1)
            self.fit(chunk_dataset, epochs = self.__epochs)

    def predict_single(self, text: str):
        text_chunks = tp.preprocess_text(
            text,
            chunk_size = self.__chunk_size,
            chunk_overlap = self.__chunk_overlap
        )

        if len(text_chunks) == 0:
            return -1

        chunks_features = self.__tokenizer(
            text_chunks,
            padding = "max_length",
            truncation = True,
            max_length = self.__max_length,
            return_tensors = "tf"
        )
        chunk_predictions = []
        for i in range(len(chunks_features["input_ids"])):
            chunk_features = {
                "input_ids": tf.expand_dims(chunks_features["input_ids"][i], 0),
                "attention_mask": tf.expand_dims(chunks_features["attention_mask"][i], 0),
                "token_type_ids": tf.expand_dims(chunks_features["token_type_ids"][i], 0)
            }
            chunk_dataset = tf.data.Dataset.from_tensor_slices(dict(chunk_features))
            chunk_dataset = chunk_dataset.batch(1)
            chunk_prediction = self.predict(chunk_dataset)
            chunk_predictions.append(int(round(chunk_prediction[0][0])))

        num_ones = sum(chunk_predictions)
        majority_label = 1 if num_ones >= len(chunk_predictions) / 2 else 0
        return majority_label

        
