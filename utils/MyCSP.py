from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np


class MyCSP(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=4):
        self.n_components = n_components

    def reshape_data(self, class_data):
        epochs, channels, time_points = class_data.shape

        class_data_2d = np.reshape(class_data, (epochs * channels,
                                                time_points))

        return class_data_2d

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=False, allow_nd=True, accept_sparse=True)

        # Set n_features for validation on transform
        self.n_features_ = X.shape[1]

        # Implement Common Spatial Pattern algorithm

        class_1_data = X[y == 1]
        class_2_data = X[y == 2]

        # Reshape the data into a 2D array
        class_1_data = self.reshape_data(class_1_data)
        class_2_data = self.reshape_data(class_2_data)

        # Calculate the covariance matrices for each class
        cov_class_1 = np.cov(class_1_data, rowvar=False)
        cov_class_2 = np.cov(class_2_data, rowvar=False)

        # Calculate the joint covariance matrix
        cov_joint = cov_class_1 + cov_class_2

        # Calculate the eigenvalues and eigenvectors of the
        # joint covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_joint)

        # Sort the eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_indices = sorted_indices[:self.n_components]

        # Get the eigenvectors corresponding to the top eigenvalues
        csp_filters = eigenvectors[:, top_indices]

        self.filters_ = csp_filters

        return self

    def transform(self, X):
        # Check if fit has been called
        check_is_fitted(self, 'n_features_')
        # Input validation
        X = check_array(X, allow_nd=True, ensure_2d=False, accept_sparse=True)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        # Apply the CSP filters to the data
        transformed = []
        for trial in X:
            transformed_trial = np.dot(trial, self.filters_)
            transformed.append(transformed_trial)

        transformed = np.array(transformed)

        # Reshape the transformed data into a 2D array
        transformed = transformed.reshape(transformed.shape[0], -1)

        return transformed
