import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir = ".", pattern = "unit_tests.py")
    runner = unittest.TextTestRunner()
    runner.run(suite)
