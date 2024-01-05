import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

class TextProcessor:
    @staticmethod
    def preprocess_text(text: str, chunk_size = 512, chunk_overlap = 64) -> list[list[str]]:
        tokens = TextProcessor.tokenize_text(text)
        token_chunks = TextProcessor.divide_into_chunks(tokens, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        strings = TextProcessor.convert_token_chunks_to_strings(token_chunks)
        return strings
        
    # TODO - think about tokenizing dashes, when they are not a part of a compound adjective (like well-known)
    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        cleaned_text = text.replace("\n", " ")
        tokens = nlp(cleaned_text)
        return [token.text for token in tokens if not token.is_space]
    
    @staticmethod
    def divide_into_chunks(tokens: list[str], chunk_size: int = 512, chunk_overlap: int = 64) -> list[list[str]]:
        if len(tokens) == 0:
            return []
        
        chunks = []
        if chunk_overlap > chunk_size:
            chunk_overlap = 1

        step = chunk_size - chunk_overlap
        if step <= 0:
            return []
        
        for i in range(0, len(tokens) - chunk_overlap, step):
            chunk = tokens[i:i + chunk_size]
            chunks.append(chunk)

        return chunks

    @staticmethod
    def convert_token_chunks_to_strings(token_chunks: list[list[str]]) -> list[str]:
        if len(token_chunks) == 0:
            return []
        
        strings = []
        for chunk in token_chunks:
            string = ""
            for i, token in enumerate(chunk):
                if token in {".", ",", ":", ";", "!", "?", "-"}:
                    string = string.rstrip()
                elif i > 0 and chunk[i - 1] != "-":
                    string += " "
                string += token
            string = string.rstrip()
            strings.append(string)

        return strings

