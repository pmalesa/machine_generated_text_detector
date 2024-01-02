import unittest

from text_processing.text_processor import TextProcessor as tp

class UnitTests(unittest.TestCase):
    def test_tokenize_text(self):
        # Test Case 1: Basic case
        text = "Keep tests small and focused on one aspect of your code."
        target_tokens = ["Keep", "tests", "small", "and", "focused", "on", "one", "aspect", "of", "your", "code", "."]
        tokens = tp.tokenize_text(text)
        self.assertEqual(tokens, target_tokens)

        # Test Case 2: Basic case with new line characters
        text = "This\nis\na\ntext\nwritten\nvertically\n."
        target_tokens = ["This", "is", "a", "text", "written", "vertically", "."]
        tokens = tp.tokenize_text(text)
        self.assertEqual(tokens, target_tokens)

        # Test Case 3: Basic case with double new line characters
        text = "This\n\nis\n\na\n\ntext\n\nwritten\n\nvertically\n\nwith\n\ndouble\n\nnew\n\nline\n\nsigns\n\n."
        target_tokens = ["This", "is", "a", "text", "written", "vertically", "with", "double", "new", "line", "signs", "."]
        tokens = tp.tokenize_text(text)
        self.assertEqual(tokens, target_tokens)

    def test_divide_into_chunks(self):
        # Test Case 1: Basic case
        tokens = ["Keep", "tests", "small", "and", "focused", "on", "one", "aspect", "of", "your", "code", "."]
        token_chunks = tp.divide_into_chunks(tokens, 5, 3)
        target_token_chunks = [ ["Keep", "tests", "small", "and", "focused"],
                                ["small", "and", "focused", "on", "one"],
                                ["focused", "on", "one", "aspect", "of"],
                                ["one", "aspect", "of", "your", "code"],
                                ["of", "your", "code", "."] ]
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 2: Basic case with different input
        tokens = ["This", "is", "a", "different", "set", "of", "tokens", "for", "testing", "the", "function"]
        token_chunks = tp.divide_into_chunks(tokens, 4, 2)
        target_token_chunks = [ ["This", "is", "a", "different"],
                                ["a", "different", "set", "of"],
                                ["set", "of", "tokens", "for"], 
                                ["tokens", "for", "testing", "the"], 
                                ["testing", "the", "function"] ]
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 3: No overlap
        tokens = ["Example", "with", "no", "overlap"]
        token_chunks = tp.divide_into_chunks(tokens, 2, 0)
        target_token_chunks = [ ["Example", "with"], ["no", "overlap"] ]
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 4: Full overlap
        tokens = ["Full", "overlap", "test"]
        token_chunks = tp.divide_into_chunks(tokens, 2, 1)
        target_token_chunks = [ ["Full", "overlap"], ["overlap", "test"] ]
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 5: Single token chunks
        tokens = ["One", "token", "per", "chunk"]
        token_chunks = tp.divide_into_chunks(tokens, 1, 0)
        target_token_chunks = [ ["One"], ["token"], ["per"], ["chunk"] ]
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 6: Edge case - empty token list
        tokens = []
        token_chunks = tp.divide_into_chunks(tokens, 3, 1)
        target_token_chunks = []
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 7: Chunk size larger than token list
        tokens = ["Small", "list"]
        token_chunks = tp.divide_into_chunks(tokens, 5, 1)
        target_token_chunks = [ ["Small", "list"] ]
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 8: Overlap larger than chunk size
        tokens = ["Unusual", "case", "overlap", "larger"]
        token_chunks = tp.divide_into_chunks(tokens, 2, 3)
        target_token_chunks = [ ["Unusual", "case"], ["case", "overlap"], ["overlap", "larger"] ]
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 9: Chunk size as zero
        tokens = ["Zero", "chunk", "or", "overlap"]
        token_chunks = tp.divide_into_chunks(tokens, 0, 2)
        target_token_chunks = []  # Assuming function handles zero chunk size gracefully
        self.assertEqual(token_chunks, target_token_chunks)

        # Test Case 10: All tokens in a single chunk
        tokens = ["All", "in", "one"]
        token_chunks = tp.divide_into_chunks(tokens, 3, 1)
        target_token_chunks = [ ["All", "in", "one"] ]
        self.assertEqual(token_chunks, target_token_chunks)

    def test_convert_token_chunks_to_strings(self):
        # Test Case 1: Basic case
        token_chunks = [ ["Keep", "tests", "small", "and", "focused"],
                         ["small", "and", "focused", "on", "one"],
                         ["focused", "on", "one", "aspect", "of"],
                         ["one", "aspect", "of", "your", "code"],
                         ["of", "your", "code", "."] ]
        target_strings = ["Keep tests small and focused",
                          "small and focused on one",
                          "focused on one aspect of",
                          "one aspect of your code",
                          "of your code." ]
        strings = tp.convert_token_chunks_to_strings(token_chunks)
        self.assertEqual(strings, target_strings)

        # Test Case 2: Basic case with different input
        token_chunks = [ ["This", "is", "a", "different"],
                         ["a", "different", "set", "of"],
                         ["set", "of", "tokens", ",", "for"], 
                         ["tokens", ",", "for", "testing", "the"], 
                         ["testing", "the", "function", "."] ]
        target_strings = ["This is a different",
                          "a different set of",
                          "set of tokens, for", 
                          "tokens, for testing the", 
                          "testing the function." ]
        strings = tp.convert_token_chunks_to_strings(token_chunks)
        self.assertEqual(strings, target_strings)

        # Test Case 3: Basic case with different input
        token_chunks = [ ["Example", "with"], ["no", "overlap"] ]
        target_strings = ["Example with", "no overlap"]
        strings = tp.convert_token_chunks_to_strings(token_chunks)
        self.assertEqual(strings, target_strings)

        # Test Case 4: Single token chunks
        token_chunks = [ ["One"], ["token"], ["per"], ["chunk"] ]
        target_strings = ["One", "token", "per", "chunk"]
        strings = tp.convert_token_chunks_to_strings(token_chunks)
        self.assertEqual(strings, target_strings)

        # Test Case 5: Edge case - empty list
        token_chunks = []
        target_strings = []
        strings = tp.convert_token_chunks_to_strings(token_chunks)
        self.assertEqual(strings, target_strings)    

        # Test Case 6: Basic case with input containing many different punctuation characters
        token_chunks = [ ["This", "example", ",", "however", ",", "is", "a", "little", "different", "!", "Ha", "ha", "!"],
                         ["Do", "You", "want", "to", "know", "why", "?", "Let", "me", "tell", "You", ":"], 
                         ["First", "of", "all", ",", "it", "has", "many", "different", "punctuation", "characters", ";", 
                          "Secondly", ",", "I", "don't", "know", "what", "else", "to", "write", "in", "this", "test", "case" ";",
                          "Thirdly", ",", "try", "this", "!"],
                         ["User-friendly", ",", "well-known", ",", "up-to-date", ",", "state-of-the-art", "!"] ]
        target_strings = ["This example, however, is a little different! Ha ha!",
                          "Do You want to know why? Let me tell You:",
                          "First of all, it has many different punctuation characters; Secondly, I don't know what else to write in this test case; Thirdly, try this!",
                          "User-friendly, well-known, up-to-date, state-of-the-art!" ]
        strings = tp.convert_token_chunks_to_strings(token_chunks)
        self.assertEqual(strings, target_strings)    

    def test_preprocess_text(self):
        input_data_file_path = "data/unit_tests/preprocess_text_input.txt"
        try:
            with open(input_data_file_path, "r") as file:
                text_data = file.read()
        except IOError as e:
            print(f"Error reading {input_data_file_path}: {e}")

        chunk_file_paths = ["data/unit_tests/chunk_1.txt", "data/unit_tests/chunk_2.txt",
                            "data/unit_tests/chunk_3.txt", "data/unit_tests/chunk_4.txt",
                            "data/unit_tests/chunk_5.txt"]
        
        chunks = []
        for chunk_file_path in chunk_file_paths:
            try:
                with open(chunk_file_path, "r") as file:
                    chunks.append(file.read())
            except IOError as e:
                print(f"Error reading {chunk_file_path}: {e}")

        strings = tp.preprocess_text(text_data)
        for i, chunk in enumerate(chunks):
            self.assertEqual(strings[i], chunk)

if __name__ == "__main__":
    unittest.main()