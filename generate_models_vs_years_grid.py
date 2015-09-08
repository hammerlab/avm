import argparse

import data
from models import hyperparameter_grid
parser = argparse.ArgumentParser()
parser.add_argument("--plot-file",
    default="plot.png")


parser.add_argument("--min-obliteration-years",
    default=2,
    type=int,
    help="Smallest value for binarizing the obliteration target")


parser.add_argument("--max-obliteration-years",
    default=5,
    type=int,
    help="Largest value for binarizing the obliteration target")

if __name__ == "__main__":
    args = parser.parse_arg()

    # consider each output target of years until AVM obliteration
    for obliteration_years in range(args.min_obliteration_years, args.max_obliteration_years + 1):
        dataset, full_df = data.load_datasets(
            obliteration_years=args.obliteration_years)
        X = dataset["X"]
        Y = dataset["Y"]
        assert len(X) == len(Y)
        VRAS = dataset["VRAS"]
        FP = dataset["FP"]
        SM = dataset["SM"]
        Y_binary = Y == 2
        grid = hyperparameter_grid(
            logistic_regression=True,
            rbf_svm=True,
            gradient_boosting=True,
            extra_trees=True,
            random_forest=True)
