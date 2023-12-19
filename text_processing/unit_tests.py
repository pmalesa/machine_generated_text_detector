import unittest

from text_processing.text_processor import TextProcessor as tp

class UnitTests(unittest.TestCase):
    def test_tokenize_text(self):
        text = "Keep tests small and focused on one aspect of your code."
        target_tokens = ["Keep", "tests", "small", "and", "focused", "on", "one", "aspect", "of", "your", "code", "."]
        tokens = tp.tokenize_text(text)
        self.assertEqual(tokens, target_tokens)

        text = "This\nis\na\ntext\nwritten\nvertically\n."
        target_tokens = ["This", "is", "a", "text", "written", "vertically", "."]
        tokens = tp.tokenize_text(text)
        self.assertEqual(tokens, target_tokens)

        text = "This\n\nis\n\na\n\ntext\n\nwritten\n\nvertically\n\nwith\n\ndouble\n\nnew\n\nline\n\nsigns\n\n."
        target_tokens = ["This", "is", "a", "text", "written", "vertically", "with", "double", "new", "line", "signs", "."]
        tokens = tp.tokenize_text(text)
        self.assertEqual(tokens, target_tokens)


    def test_divide_into_chunks(self):
        tokens = ["Keep", "tests", "small", "and", "focused", "on", "one", "aspect", "of", "your", "code", "."]
        token_chunks = tp.divide_into_chunks(tokens, 5, 3)
        target_token_chunks = [ ["Keep", "tests", "small", "and", "focused"],
                                ["small", "and", "focused", "on", "one"],
                                ["focused", "on", "one", "aspect", "of"],
                                ["one", "aspect", "of", "your", "code"],
                                ["of", "your", "code", "."] ]
        self.assertEqual(token_chunks, target_token_chunks)

    def test_convert_token_chunks_to_strings(self):
        pass

    def test_preprocess_text(self):
        pass











if __name__ == "__main__":
    unittest.main()