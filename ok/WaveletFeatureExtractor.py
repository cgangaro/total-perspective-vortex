from scipy.signal import cwt, morlet
import numpy as np

class WaveletFeatureExtractor:
    def __init__(self, wavelet='morl', scales=np.arange(1, 32), mode='magnitude'):
        """
        WaveletFeatureExtractor applique la transformation par ondelettes continues (CWT) 
        et extrait des caractéristiques.
        
        :param wavelet: Type d'ondelette (ex. 'morl').
        :param scales: Échelles pour CWT.
        :param mode: Comment traiter les coefficients complexes. Options : 'magnitude', 'real', 'imag'.
        """
        self.wavelet = wavelet
        self.scales = scales
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Applique la transformation par ondelettes sur des signaux multicanaux.
        
        :param X: ndarray, forme (n_samples, n_channels, n_times)
        :return: ndarray, caractéristiques extraites
        """
        n_samples, n_channels, n_times = X.shape
        features = []
        for sample in X:
            coeffs = []
            for channel in sample:
                cwt_coeffs = cwt(channel, morlet, self.scales)

                # Traitement des données complexes
                if self.mode == 'magnitude':
                    processed_coeffs = np.abs(cwt_coeffs)
                elif self.mode == 'real':
                    processed_coeffs = np.real(cwt_coeffs)
                elif self.mode == 'imag':
                    processed_coeffs = np.imag(cwt_coeffs)
                else:
                    raise ValueError("Mode must be 'magnitude', 'real', or 'imag'.")

                # Extraire l'énergie moyenne pour chaque échelle
                energy = np.mean(processed_coeffs ** 2, axis=1)
                coeffs.extend(energy)

            features.append(coeffs)
        return np.array(features)
