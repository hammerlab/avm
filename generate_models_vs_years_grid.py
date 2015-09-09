import argparse
from collections import OrderedDict

import numpy as np
import seaborn

import data
from models import hyperparameter_grid
from cv import find_best_model

import matplotlib
matplotlib.use('qt4agg')  # Can also use 'tkagg' or 'webagg'

parser = argparse.ArgumentParser()
parser.add_argument("--plot-file",
    default="plot.png")

parser.add_argument("--logistic-regression",
    default=False,
    action="store_true")

parser.add_argument("--random-forest",
    default=False,
    action="store_true")

parser.add_argument("--extra-trees",
    default=False,
    action="store_true")

parser.add_argument("--svm",
    default=False,
    action="store_true")

parser.add_argument("--gradient-boosting", default=False, action="store_true")

parser.add_argument("--min-obliteration-years",
    default=2,
    type=int,
    help="Smallest value for binarizing the obliteration target")


parser.add_argument("--max-obliteration-years",
    default=5,
    type=int,
    help="Largest value for binarizing the obliteration target")

parser.add_argument("--output-file", default=None, help="Output figure to this file")

parser.add_argument(
        "--num-samples",
        default=None,
        type=int,
        help="Ignore all samples after this number is loaded")

if __name__ == "__main__":
    args = parser.parse_args()

    year_to_index = OrderedDict()
    model_to_index = OrderedDict()

    # dictionary from "years to obliteration" to number of samples
    dataset_sizes = {}
    # mapping from (year, model name) pairs to AUC
    aucs_dict = {}
    # consider each output target of years until AVM obliteration
    for obliteration_years in reversed(
            range(args.min_obliteration_years, args.max_obliteration_years + 1)):
        print("\n### Obliteration Years = %d\n\n" % obliteration_years)
        dataset, full_df = data.load_datasets(
            obliteration_years=obliteration_years)
        X = dataset["X"][:args.num_samples]
        Y = dataset["Y"][:args.num_samples]
        dataset_sizes[obliteration_years] = len(X)
        assert len(X) == len(Y)
        VRAS = dataset["VRAS"][:args.num_samples]
        FP = dataset["FP"][:args.num_samples]
        SM = dataset["SM"][:args.num_samples]
        Y_binary = Y == 2
        grid = hyperparameter_grid(
            logistic_regression=args.logistic_regression,
            svm=args.svm,
            gradient_boosting=args.gradient_boosting,
            extra_trees=args.extra_trees,
            random_forest=args.random_forest)
        if len(grid) == 0:
            raise ValueError("No prediction models selected")
        best_overall_results, results_per_model_class = find_best_model(
            X, Y_binary, grid)

        year_to_index[obliteration_years] = len(year_to_index)
        for (model_class_name, model_results) in results_per_model_class.items():
            if model_class_name not in model_to_index:
                model_to_index[model_class_name] = len(model_to_index)
            aucs_dict[(obliteration_years, model_class_name)] = model_results.auc
    # make a rectangular array for the heatmap
    n_years = len(year_to_index)
    n_models = len(model_to_index)
    aucs_array = np.zeros((n_years, n_models), dtype=float)
    for (year, model_class_name), auc in aucs_dict.items():
        year_index = year_to_index[year]
        model_index = model_to_index[model_class_name]
        aucs_array[year_index, model_index] = auc

    model_labels = list(model_to_index.keys())
    model_labels = [x.replace("Classifier", "") for x in model_labels]

    year_labels = list(year_to_index.keys())
    year_labels = [
        "%d (n=%d)" % (year, dataset_sizes[year])
    ]
    heatmap = seaborn.heatmap(
        data=aucs_array,
        xticklabels=model_labels,
        yticklabels=year_labels,
        linewidths=2.0)
    heatmap.set_xlabel("Model")
    heatmap.set_ylabel("Target Years Until AVM Obliteration")
    if args.output_file:
        heatmap.get_figure().savefig(args.output_file)
