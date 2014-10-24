
import itertools

from sklearn.metrics import roc_auc_score

def class_prob(model, X):
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)
        return prob[:, -1]
    else:
        pred = model.decision_function(X)
        if len(pred.shape) > 1 and pred.shape[1] ==1:
            pred = pred[:, 0]
        assert len(pred.shape) == 1, pred.shape
        return pred

class Normalizer(object):
    """
    Subtract mean and divide features by standard deviation
    before fitting/predicting
    """
    def __init__(self, model, Xm = None, Xs = None):
        self.model = model
        self.Xm = Xm
        self.Xs = Xs

    def fit(self, X, y, *args, **kwargs):
        self.Xm = X.mean(axis=0)
        X = X - self.Xm
        self.Xs = X.std(axis=0)
        self.Xs[self.Xs == 0] = 1
        X = X / self.Xs
        self.model.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        X = X - self.Xm
        X /= self.Xs
        return self.model.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        X = X - self.Xm
        X /= self.Xs
        return self.model.predict_proba(X, *args, **kwargs)

    def decision_function(self, X, *args, **kwargs):
        X = X - self.Xm
        X /= self.Xs
        return self.model.decision_function(X, *args, **kwargs)

    def get_params(self, deep=False):
        return { 'Xm' : self.Xm, 'Xs': self.Xs, 'model' : self.model }

def roc_auc(model, X, y):
    p = class_prob(model, X)
    return roc_auc_score(y, p)

def normalize(X_train, X_test):
    Xm = X_train.mean(axis=0)
    X_train = X_train - Xm
    X_test = X_test - Xm
    Xs = X_train.std(axis=0)
    Xs[Xs==0] = 1
    X_train /= Xs
    X_test /= Xs
    return X_train, X_test

def all_combinations(param_grid):
    return [
                {key: value for (key, value) in zip(param_grid, values)}
                for values
                in itertools.product(*param_grid.values())
    ]
