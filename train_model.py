import argparse

from training_module.training_module import TrainingModule

def run(model: str):
    print(f"Chosen model: {model}")
    tm = TrainingModule()

    if (model == "gtd"):
        tm.train_gtd()
    elif (model == "lmd"):
        tm.train_lmd()
    else:
        tm.train_tsd()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the script by choosing the specific model to train.")

    parser.add_argument("model", choices = ["gtd", "lmd", "tsd"], help = "Model to run the script with. Options are \"gtd\", \"lmd\", or \"tsd\")")

    args = parser.parse_args()

    run(args.model)