from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np


class CSP(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=4):
        self.n_components = n_components

    def reshape_data(self, class_data):
        epochs, channels, timePoints = class_data.shape
        classData2D = np.reshape(class_data, (epochs * channels, timePoints))
        # (epochs, channels, time points) to (epochs*channels, time points)
        # each row represents a temporal data for a channel in a specific epoch
        # each column represents a temporal point,
        # common for all channels and epochs
        return classData2D

    def fit(self, X, y=None):
        # X shape = epochs x channels x time points
        # Example: (10 epochs, 64 channels, 100 time points) means 100 measure
        # points for 64 channels for 10 epochs
        X = check_array(X, ensure_2d=False, allow_nd=True, accept_sparse=True)
        # check_array check if the input is in the right format and
        # standardize it
        # ensure_2d = True means that the input should be 2D
        # allow_nd = True means that the input can be nD (ex: 3D)
        # accept_sparse = True means that the input can be sparse
        # (with lots of zeros)

        self.n_features_ = X.shape[1]
        # Save the number of features in the input data.
        # Here, the number of features is the number of channels

        class_1_data = X[y == 1]
        class_2_data = X[y == 2]
        # Separate the data into two classes

        class_1_data = self.reshape_data(class_1_data)
        class_2_data = self.reshape_data(class_2_data)
        # Reshape the data into a 2D array

        cov_class_1 = np.cov(class_1_data, rowvar=False)
        cov_class_2 = np.cov(class_2_data, rowvar=False)
        # a positive covariance means that the two variables increase
        # or decrease together
        # a negative covariance means that one variable increases while
        # the other decreases
        # a covariance of 0 means that the variables are independent
        # Here, the result is a covariance matrix channel x channel
        # It represents the relationship between the channels

        cov_joint = cov_class_1 + cov_class_2
        # Calculate the joint covariance matrix

        eigenValues, eigenVectors = np.linalg.eigh(cov_joint)
        # a eigen value measures the amount of variance in the data,
        # so the importance of the corresponding eigen vector
        # a eigen vector is a direction in the data space

        sortedEigenValues = np.argsort(eigenValues)[::-1]
        topEigenValues = sortedEigenValues[:self.n_components]
        # topEigenValues contains the indexes of the top eigen values

        cspFilters = eigenVectors[:, topEigenValues]
        # cspFilters contains the top eigen vectors

        self.filters_ = cspFilters
        return self

    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        # Check if the model has been fitted

        X = check_array(X, allow_nd=True, ensure_2d=False, accept_sparse=True)

        if X.shape[1] != self.n_features_:
            raise ValueError('Input data has a different number of features' +
                             'than the training data')
        transformed = []
        for trial in X:
            transformed_trial = np.dot(trial, self.filters_)
            transformed.append(transformed_trial)

        transformed = np.array(transformed)

        transformed = transformed.reshape(transformed.shape[0], -1)

        return transformed
