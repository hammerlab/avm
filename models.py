import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

def log_spaced_range(min_value, max_value, n_values):
    log_min_value = np.log10(min_value)
    log_max_value = np.log10(max_value)
    return 10.0 ** np.linspace(log_min_value, log_max_value, n_values)

def hyperparameter_grid(
        logistic_regression=False,
        svm=False,
        random_forest=False,
        extra_trees=False,
        gradient_boosting=False,
        min_float_value=10.0**-4.0,
        max_float_value=100.0,
        number_float_values=12,
        min_small_int=2,
        max_small_int=4):
    grid = {}
    if logistic_regression:
        grid[LogisticRegression] = {
            'penalty': ["l1"],
            'fit_intercept': [False, True],
            'C': log_spaced_range(min_float_value, max_float_value, number_float_values),
        }

    if svm:
        # since we're sweeping over both C and gamma,
        # want the total number of grid entries to be approximately
        # the same as LogisticRegression, so reducing number of
        # candidate values attempted
        grid[LinearSVC] = {
            'C': log_spaced_range(min_float_value, max_float_value, number_float_values),
        }
    if random_forest:
        grid[RandomForestClassifier] = {
            'criterion': ["gini"],
            'max_depth': [10, 20, 30],
            'n_estimators': [25, 50, 100],
            'min_samples_split': list(range(min_small_int, max_small_int + 1)),
        }
    if extra_trees:
        grid[ExtraTreesClassifier] = {
            'criterion': ["gini"],
            'max_depth': [10, 20, 30],
            'n_estimators': [25, 50, 100],
            'min_samples_split': list(range(min_small_int, max_small_int + 1)),
        }
    if gradient_boosting:
        grid[GradientBoostingClassifier] = {
            'learning_rate': log_spaced_range(min_float_value, 1.0, number_float_values),
            'subsample': [0.5, 1.0],
            'n_estimators': [25, 50, 100],
            'max_depth': list(range(min_small_int, max_small_int + 1)),
        }
    return grid
