from __future__ import print_function, division, absolute_import
import itertools

import numpy as np
from sklearn.metrics import roc_auc_score

def class_prob(model, X):
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)
        return prob[:, -1]
    else:
        pred = model.decision_function(X)
        if len(pred.shape) > 1 and pred.shape[1] == 1:
            pred = pred[:, 0]
        assert len(pred.shape) == 1, pred.shape
        return pred

def roc_auc(model, X, y):
    p = class_prob(model, X)
    return roc_auc_score(y, p)

def normalize(X_train, X_test):
    Xm = X_train.mean(axis=0)
    X_train = X_train - Xm
    X_test = X_test - Xm
    Xs = X_train.std(axis=0)
    Xs[Xs == 0] = 1
    X_train /= Xs
    X_test /= Xs
    return X_train, X_test

def all_combinations(param_grid):
    keys = []
    value_lists = []
    for (key, value_list) in param_grid.items():
        keys.append(key)
        value_lists.append(value_list)
    return [
                {key: value for (key, value) in zip(keys, values)}
                for values
                in itertools.product(*value_lists)
    ]

def cv_indices_generator(n_samples, n_iters, sample_with_replacement=False):
    """
    Generator returns (iteration, (train_indices, test_indices))
    """
    for i in range(n_iters):
        n_train = 2 * n_samples // 3
        if sample_with_replacement:
            # bootstrap sampling training sets which are 2/3 of the full data
            train_indices = np.random.randint(0, n_samples, n_train)
            train_indices_set = set(train_indices)
            test_indices = np.array([i for i in range(n_samples) if i not in train_indices_set])
        else:
            all_indices = np.arange(n_samples)
            np.random.shuffle(all_indices)
            train_indices = all_indices[:n_train]
            test_indices = all_indices[n_train:]
        print("# total = %d, # train = %d, # test = %d, max train index = %d" % (
            n_samples, len(train_indices), len(test_indices), max(train_indices)))
        assert len(train_indices) + len(test_indices) == n_samples
        yield (i, (train_indices, test_indices))
