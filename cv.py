from __future__ import division, print_function, absolute_import

from sklearn.cross_validation import KFold
import numpy as np
from helpers import Normalizer, roc_auc, all_combinations

def find_best_model(
        X, Y, model_class, param_grids,
        n_folds=5,
        target_value=1,
        normalize_features=True):
    curr_best_auc = 0
    curr_best_model = None
    curr_best_params = None

    for i, param_grid in enumerate(param_grids):
        print("Param_grid #%d/%d" % (i + 1, len(param_grids)), param_grid)
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
            if curr_auc > curr_best_auc:
                curr_best_auc = curr_auc
                curr_best_model = normalized
                curr_best_params = param_combination
    print("== Best Model: %s, AUC = %0.4f\n" % (curr_best_model, curr_best_auc))
    return curr_best_model, curr_best_params, curr_best_auc
