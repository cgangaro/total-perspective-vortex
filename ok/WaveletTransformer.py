from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pywt
import numpy as np


class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X est de forme (n_samples, n_channels, n_times)
        n_samples, n_channels, n_times = X.shape
        features = []
        for sample in X:
            coeffs = []
            for channel_data in sample:
                # Effectuer la décomposition par ondelettes
                coeffs_channel = pywt.wavedec(channel_data, self.wavelet, level=self.level)
                # Extraire des caractéristiques, par exemple l'énergie des coefficients
                coeffs_channel_energy = [np.sum(np.square(c)) for c in coeffs_channel]
                coeffs.extend(coeffs_channel_energy)
            features.append(coeffs)
        return np.array(features)
