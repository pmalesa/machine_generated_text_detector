import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

class TextProcessor:
    @staticmethod
    def preprocess_text(text: str) -> list[list[str]]:
        tokens = TextProcessor.tokenize_text(text)
        

    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        cleaned_text = text.replace("\n", " ")
        tokens = nlp(cleaned_text)
        return [token.text for token in tokens if not token.is_space]
    
    @staticmethod
    def divide_into_chunks(tokens: list[str], chunk_size: int = 512, chunk_overlap: int = 64) -> list[list[str]]:
        pass

    @staticmethod
    def convert_token_chunks_to_strings(token_chunks: list[list[str]]) -> list[str]:
        pass

    @staticmethod
    def create_embedding(text: str) -> np.ndarray:
        pass
