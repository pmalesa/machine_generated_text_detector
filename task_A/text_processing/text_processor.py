import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

class TextProcessor:

    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        doc = nlp(text)
        return [token.text for token in doc]

    @staticmethod
    def create_embedding(text: str) -> np.ndarray:
        pass
