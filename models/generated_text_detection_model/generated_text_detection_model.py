from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

from text_processing.text_processor import TextProcessor as tp

tf.config.run_functions_eagerly(True)

class GeneratedTextDetectionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.__tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.__deberta_model = TFAutoModel.from_pretrained("microsoft/deberta-base")
        self.__deberta_model.trainable = False
        self.__intermediate_dense_layer = tf.keras.layers.Dense(768, activation = "relu")
        self.__output_dense_layer = tf.keras.layers.Dense(1, activation = "softmax")

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training = False):
        outputs = self.__deberta_model(inputs)
        cls_output = outputs[0][:, 0, :]
        intermediate_output = self.__intermediate_dense_layer(cls_output)
        return self.__output_dense_layer(intermediate_output)

    def train(self, texts: list[str], labels: list[int], learning_rate = 0.001, epochs = 10, batch_size = 16, chunk_size = 128, chunk_overlap = 32):
        if len(texts) != len(labels):
            print("[ERROR] Number of examples is not equal to the number of labels!")
            return
        
        self.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
            loss = tf.keras.losses.BinaryCrossentropy(),
            metrics = [tf.keras.metrics.BinaryAccuracy()]
        )

        for i, (text, label) in enumerate(zip(texts, labels)):
            self.__train_single(text, label, epochs, batch_size, chunk_size, chunk_overlap)

    def __train_single(self, text: str, label: int, epochs: int, batch_size: int, chunk_size: int, chunk_overlap: int):
        text_chunks = tp.preprocess_text(text, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        train_features = self.__tokenizer(text_chunks, padding = True, truncation = True, return_tensors = "tf")
        train_labels = [label] * len(text_chunks)
        train_labels_tensor = tf.constant(train_labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_features), train_labels_tensor))
        train_dataset = train_dataset.batch(batch_size)
        self.fit(train_dataset, epochs = epochs)
