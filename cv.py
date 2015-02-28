from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
import numpy as np
from helpers import Normalizer, class_prob, roc_auc, all_combinations

def find_best_model(X, Y, model_class, param_grids, n_folds=5):
    curr_best_auc = 0
    curr_best_model = None
    curr_best_params = None

    for i, param_grid in enumerate(param_grids):
        print "Param_grid #%d/%d" % (i+1, len(param_grids)), param_grid
        for param_combination in all_combinations(param_grid):
            for k,v in param_combination.iteritems():
                assert type(v) != list, "Can't give a list of parameters for %s" % k
            curr_model = model_class(**param_combination)
            print "--", curr_model

            normalized = Normalizer(curr_model)
            # nested cross validation over just the current training set
            # to evaluate each parameter set's performance
            nested_kfold = KFold(len(Y), n_folds, shuffle=False)
            curr_aucs = []
            for (nested_train_idx, nested_test_idx) in nested_kfold:
                normalized.fit(X[nested_train_idx, :], Y[nested_train_idx])
                curr_auc = roc_auc(normalized, X[nested_test_idx, :], Y[nested_test_idx])
                curr_aucs.append(curr_auc)
            curr_auc = np.mean(curr_aucs)
            print "-- AUC: %0.4f" % curr_auc
            print
            if curr_auc > curr_best_auc:
                curr_best_auc = curr_auc
                curr_best_model = normalized
                curr_best_params = param_combination
    print "== Best Model: %s, AUC = %0.4f" % (curr_best_model.model, curr_best_auc)
    print
    return curr_best_model, curr_best_params, curr_best_auc

def evaluate(X, Y, candidate_models, n_folds=5):
    n_samples, n_features = X.shape
    assert len(Y) == n_samples

    cv_aucs = {}
    kfold = KFold(n_samples, n_folds, shuffle=True)
    for fold_idx, (train_idx, test_idx) in enumerate(kfold):
        n_cv_train = len(train_idx)
        n_cv_test = len(test_idx)

        print "Fold #%d" % (fold_idx+1)
        print "# total train", n_cv_train
        print "# test", n_cv_test
        print

        X_train = X[train_idx, :]
        Y_train = Y[train_idx]
        X_test = X[test_idx, :]
        Y_test = Y[test_idx]

        # dictionary from model class names to pair of (best_hyperparam_model, params_dict)
        curr_fold_models = {}
        for model, param_grids in candidate_models.iteritems():

            if not isinstance(param_grids, list):
                assert isinstance(param_grids, dict), type(param_grids)
                param_grids = [param_grids]
            name = model.__class__.__name__
            print "Base Model", model
            curr_best_model, curr_best_params, _ = find_best_model(X_train, Y_train, model.__class__, param_grids)

            # have to select the '.model' field since these are all Normalized objects
            curr_fold_models[name] = (curr_best_model, curr_best_params)
        print
        print "Done with hyperparameter selection"
        print
        print "=========="
        for k,(best_model, best_params) in curr_fold_models.iteritems():
            print k
            print "-- best model:", best_model.model
            print "-- params:", best_params

            best_model.fit(X_train, Y_train)
            auc = roc_auc(best_model, X_test, Y_test)
            print "-- Fold #%d test AUC for %s: %0.4f" %(fold_idx+1, k, auc)
            if k not in cv_aucs:
                cv_aucs[k] = []
            cv_aucs[k].append(auc)
            print
    return cv_aucs
