class Normalizer(object):
    """
    Subtract mean and divide features by standard deviation
    before fitting/predicting
    """
    def __init__(self, model, Xm=None, Xs=None):
        self.model = model
        self.Xm = Xm
        self.Xs = Xs

        # only provide `predict_proba` method to normalized model
        # if it's available in the underlying model
        if hasattr(model, 'predict_proba'):
            self.predict_proba = self._predict_proba

    def __str__(self):
        return "Normalizer(%s)" % self.model

    @property
    def coef_(self):
        return self.model.coef_

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

    def _predict_proba(self, X, *args, **kwargs):
        X = X - self.Xm
        X /= self.Xs
        return self.model.predict_proba(X, *args, **kwargs)

    def decision_function(self, X, *args, **kwargs):
        X = X - self.Xm
        X /= self.Xs
        return self.model.decision_function(X, *args, **kwargs)

    def get_params(self, deep=False):
        return {'Xm': self.Xm, 'Xs': self.Xs, 'model': self.model}
