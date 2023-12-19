import argparse

from testing_module.testing_module import TestingModule

def run(model: str):
    print(f"Chosen model: {model}")
    tm = TestingModule()

    if (model == "gtd"):
        tm.test_gtd()
    elif (model == "lmd"):
        tm.test_lmd()
    else:
        tm.test_tsd()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the script by choosing the specific model to test.")

    parser.add_argument("model", choices = ["gtd", "lmd", "tsd"], help = "Model to run the script with. Options are \"gtd\", \"lmd\", or \"tsd\")")

    args = parser.parse_args()

    run(args.model)