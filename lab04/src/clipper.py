import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Learns the lower and upper percentiles on the data it is fitted on, then clips."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.percentile(X, self.lower, axis=0)
        self.high_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)