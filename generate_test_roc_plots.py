from __future__ import division, print_function, absolute_import

import argparse

# import seaborn
import sklearn
import sklearn.linear_model
import sklearn.ensemble
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pylab as plt

from helpers import roc_auc, class_prob
import data
import cv
from models import hyperparameter_grid

parser = argparse.ArgumentParser()
parser.add_argument("--plot-file",
    default="plot.png")

parser.add_argument("--coefs-file", default="coefs.txt")
parser.add_argument("--obliteration-years",
    default=4,
    type=int,
    help="Number of years to wait for obliteration")

parser.add_argument("--line-width",
    default=5,
    type=int,
    help="Line width for ROC curve plot")

parser.add_argument("--opacity",
    default=1.0,
    type=float)

parser.add_argument("--disable-feature-normalization",
    default=False,
    action="store_true",
    help="Don't subtract mean or divide by standard deviation")


def generate_roc_plot(
        X_train,
        Y_train,
        X_test,
        Y_test,
        SM_test,
        VRAS_test,
        FP_test,
        columns,
        classifier_class,
        classifier_hyperparameters,
        target_value=1,
        line_width=5,
        normalize_features=True,
        opacity=1.0,
        filename="plot.png",
        obliteration_years=5):

    axes = plt.axes()

    # use cross-validation over the bootstrap training set
    # to choose hyperparameters
    overall_best_results, results_per_class = cv.find_best_model(
        X_train,
        Y_train,
        {classifier_class: classifier_hyperparameters},
        target_value=1,
        normalize_features=normalize_features)
    best_model = overall_best_results.model
    params = overall_best_results.params

    # fit the best hyperparameter model with the full bootstrap training
    # set
    best_model.fit(X_train, Y_train)

    # print linear model coefficients for each feature
    for (col_name, coef) in zip(columns, list(best_model.coef_[0])):
        print("\t%s %s: %f" % ("-->" if coef != 0 else "   ", col_name, coef))

    print(">>> Params", params)
    auc = roc_auc(best_model, X_test, Y_test)
    print(">>> AUC", auc)

    probs = class_prob(best_model, X_test)
    fpr, tpr, _ = roc_curve(Y_test, probs)
    axes.plot(
        fpr,
        tpr,
        lw=line_width,
        alpha=opacity,
        color=(0.15, 0.25, 0.7),
        label="LR (AUC=%0.3f)" % auc)

    # need to make VRAS negative for ROC curve since it's
    # anti-correlated with good outcomes
    VRAS_fpr, VRAS_tpr, _ = roc_curve(Y_test, -VRAS_test)
    VRAS_auc = roc_auc_score(Y_test, -VRAS_test)
    axes.plot(
        VRAS_fpr,
        VRAS_tpr,
        lw=line_width,
        alpha=opacity,
        color=(0.7, 0.25, 0.1),
        label="VRAS (AUC=%0.3f)" % VRAS_auc)

    # need to make FP negative for ROC curve since it's
    # anti-correlated with good outcomes
    FP_fpr, FP_tpr, _ = roc_curve(Y_test, -FP_test)
    FP_auc = roc_auc_score(Y_test, -FP_test)

    axes.plot(
        FP_fpr,
        FP_tpr,
        lw=line_width,
        alpha=opacity,
        color=(0.1, 0.75, 0.2),
        markersize=2.0,
        label="FP (AUC=%0.3f)" % FP_auc)

    # need to make SM negative for ROC curve since it's
    # anti-correlated with good outcomes
    SM_fpr, SM_tpr, _ = roc_curve(Y_test, -SM_test)
    SM_auc = roc_auc_score(Y_test, -SM_test)
    axes.plot(
        SM_fpr,
        SM_tpr,
        lw=line_width,
        alpha=opacity,
        color=(0.7, 0.1, 0.4),
        markersize=2.0,
        label="SM (AUC=%0.3f)" % SM_auc)

    # diagonal line

    axes.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

    axes.legend(loc="lower right")
    axes.set_aspect('equal')
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    # axes.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Time Horizon = %d Years' % obliteration_years)
    figure = axes.figure
    figure.savefig(filename)
    return best_model

if __name__ == "__main__":
    args = parser.parse_args()

    train_dataset, train_df = data.load_datasets(
        filename="AVM.xlsx",
        obliteration_years=args.obliteration_years)
    test_dataset, test_df = data.load_datasets(
        filename="AVM_NYU.xlsx",
        obliteration_years=args.obliteration_years)
    X_train = train_dataset["X"]
    Y_train = train_dataset["Y"]

    X_test = test_dataset["X"]
    Y_test = test_dataset["Y"]

    test_columns = test_dataset["df_filtered"].columns
    VRAS_test = test_dataset["VRAS"]
    FP_test = test_dataset["FP"]
    SM_test = test_dataset["SM"]

    lr_hyperparameters = list(
        hyperparameter_grid(logistic_regression=True).values())[0]
    print(lr_hyperparameters)
    model = generate_roc_plot(
        X_train=X_train,
        Y_train=(Y_train == 2),
        X_test=X_test,
        Y_test=(Y_test == 2),
        SM_test=SM_test,
        VRAS_test=VRAS_test,
        FP_test=FP_test,
        columns=test_columns,
        classifier_class=sklearn.linear_model.LogisticRegression,
        classifier_hyperparameters=lr_hyperparameters,
        target_value=1,
        line_width=args.line_width,
        normalize_features=not args.disable_feature_normalization,
        filename=args.plot_file,
        opacity=args.opacity,
        obliteration_years=args.obliteration_years)
    feature_nonzero_counts = np.zeros(len(test_columns), dtype=int)
    coefs = model.coef_.flatten()
    feature_nonzero_counts[coefs[0] != 0] += 1
    with open(args.coefs_file, "w") as f:
        f.write("%s\n" % model)
        f.write("All features: %s\n" % (test_columns,))
        f.write("Feature Coefs (%d/%d)\n" % (
            sum(c != 0 for c in coefs),
            len(coefs)))
        for i in np.argsort(feature_nonzero_counts):
            feature = test_columns[i]
            coef = coefs[i]
            if coef:
                f.write("\t %30s: %0.4f\n" % (feature, coef))
