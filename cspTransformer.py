from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
import numpy as np

class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=None, log=True, cov_est='concat'):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.csp = None

    def fit(self, X, y=None):
        # Assurez-vous que X est en float64
        X = X.astype(np.float64)
        print(f"Type de X dans fit: {X.dtype}")
        self.csp = CSP(n_components=self.n_components, reg=self.reg, log=self.log, cov_est=self.cov_est)
        self.csp.fit(X, y)
        return self

    def transform(self, X):
        X = X.astype(np.float64)
        print(f"Type de X dans transform: {X.dtype}")
        return self.csp.transform(X)
