import argparse

from training_module.training_module import TrainingModule
from testing_module.testing_module import TestingModule

def run(mode: str):
    print(f"Chosen mode: {mode}")
    if (mode == "train"):
        tm = TrainingModule()
        tm.run()
    else:
        tm = TestingModule()
        tm.run()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the script in a specific mode (train or test).")

    parser.add_argument("mode", choices = ["train", "test"], help = "Mode to run the script in. Options are \"train\" or \"text\")")

    args = parser.parse_args()

    run(args.mode)