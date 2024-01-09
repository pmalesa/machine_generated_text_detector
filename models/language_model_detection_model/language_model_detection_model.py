from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import json

from text_processing.text_processor import TextProcessor as tp

tf.config.run_functions_eagerly(True)

class LanguageModelDetectionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.__tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.__deberta_model = TFAutoModel.from_pretrained("microsoft/deberta-base")
        self.__deberta_model.trainable = False
        self.__intermediate_dense_layer = tf.keras.layers.Dense(768, activation = "relu")
        self.__output_dense_layer = tf.keras.layers.Dense(6, activation = "softmax")

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training = False):
        outputs = self.__deberta_model(inputs)
        cls_output = outputs[0][:, 0, :]
        intermediate_output = self.__intermediate_dense_layer(cls_output)
        return self.__output_dense_layer(intermediate_output)

    def train(self, texts: list[str], labels: list[int], learning_rate = 0.001, epochs = 10, chunk_size = 128, chunk_overlap = 32):
        if len(texts) != len(labels):
            print("[ERROR] Number of examples is not equal to the number of labels!")
            return

        labels = tf.keras.utils.to_categorical(labels, num_classes=6)
        # Compiling the model
        self.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = [tf.keras.metrics.CategoricalAccuracy()]
        )

        # Training the model
        for i, (text, label) in enumerate(zip(texts, labels)):
            print(f"Example {i + 1}")
            self.__train_single(text, label, epochs, chunk_size, chunk_overlap)
            if i % 10 == 9:
                print(f"[INFO] Examples processed so far: {i + 1}")

    def load(self, path: str):
        self = tf.keras.models.load_model(path)

    def predict_text(self, text: str):
        pass

    def __train_single(self, text: str, label: int, epochs: int, chunk_size: int, chunk_overlap: int):
        # Divide text into chunks
        text_chunks = tp.preprocess_text(text, chunk_size = chunk_size, chunk_overlap = chunk_overlap)

        # Tokenize all chunks at once
        if len(text_chunks)!=0:
            chunks_features = self.__tokenizer(text_chunks, padding = "max_length", truncation = True, max_length = 512, return_tensors = "tf")
            for i in range(len(chunks_features["input_ids"])):
                # Extract the tokenized information for a single chunk
                chunk_features = {
                    "input_ids": tf.expand_dims(chunks_features["input_ids"][i], 0),
                    "attention_mask": tf.expand_dims(chunks_features["attention_mask"][i], 0),
                    "token_type_ids": tf.expand_dims(chunks_features["token_type_ids"][i], 0)
                }

                # Create single label tensor for the chunk
                chunk_label = tf.constant([label])

                # Create a dataset for a single chunk
                chunk_dataset = tf.data.Dataset.from_tensor_slices((dict(chunk_features), chunk_label))
                chunk_dataset = chunk_dataset.batch(1)

                self.fit(chunk_dataset, epochs = epochs)

    def __predict_single(self, text: str):
        pass
