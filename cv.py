from __future__ import division, print_function, absolute_import
from collections import namedtuple

from sklearn.cross_validation import KFold
import numpy as np
from helpers import roc_auc, all_combinations
from normalizer import Normalizer

CrossValResults = namedtuple("CrossValResults", "auc model params")

def find_best_model(
        X,
        Y,
        param_grids,
        n_folds=5,
        target_value=1,
        normalize_features=True,
        shuffle=True):
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
    auc_per_class = {}
    results_per_class = {}

    # if we were passed a single grid then turn it into a list since the
    # code below expects multiple grids

    for i, (model_class, param_grid) in enumerate(param_grids.items()):
        model_class_name = model_class.__name__
        print("Param_grid #%d/%d for %s: %s" % (
            i + 1,
            len(param_grids),
            model_class_name,
            param_grid))
        for param_combination in all_combinations(param_grid):
            for k, v in param_combination.iteritems():
                assert type(v) != list, "Can't give a list of parameters for %s" % k

            curr_model = model_class(**param_combination)
            if normalize_features:
                normalized = Normalizer(curr_model)
            else:
                normalized = curr_model
            # nested cross validation over just the current training set
            # to evaluate each parameter set's performance
            nested_kfold = KFold(len(Y), n_folds, shuffle=True)
            curr_aucs = []
            for (nested_train_idx, nested_test_idx) in nested_kfold:
                normalized.fit(X[nested_train_idx, :], Y[nested_train_idx])
                X_nested = X[nested_test_idx, :]
                Y_nested = Y[nested_test_idx]
                # skip iterations where all Y's are the same
                if (Y_nested == Y_nested[0]).all():
                    continue
                curr_auc = roc_auc(normalized, X_nested, Y_nested == target_value)
                curr_aucs.append(curr_auc)
            curr_auc = np.mean(curr_aucs)
            if curr_auc > auc_per_class.get(model_class_name, 0):
                print("-- improved %s AUC to %0.4f with %s" % (
                    model_class_name,
                    curr_auc,
                    curr_model))
                auc_per_class[model_class_name] = curr_auc
                results_per_class[model_class_name] = CrossValResults(
                    auc=curr_auc,
                    model=normalized,
                    params=param_combination)
            else:
                print("-- no improvement from %s %s, AUC = %0.4f" % (
                    model_class_name,
                    param_combination,
                    curr_auc))
    best_model_name, overall_best_results = max(
        results_per_class.items(),
        key=lambda pair: pair[1].auc)
    print("== Best Model: %s, AUC = %0.4f\n" % (
        overall_best_results.model,
        overall_best_results.auc))
    return overall_best_results, results_per_class
