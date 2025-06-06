# Fichier CSP.py
""" Reimplementation of a signal decomposition using the Common spatial pattern"""

# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(TransformerMixin, BaseEstimator):
    """
    CSP implementation based on MNE implementation

    https://github.com/mne-tools/mne-python/blob/f87be3000ce333ff9ccfddc45b47a2da7d92d69c/mne/decoding/csp.py#L565
    """
    def __init__(self, n_components=4, reg=None, log=None, cov_est='concat',
                 transform_into='average_power', norm_trace=False,
                 cov_method_params=None, rank=None,
                 component_order='mutual_info'):
        """
        Initializing the different optional parameters.
        Some checks might not be full, and all options not implemented.
        We just created the parser based on the original implementation of the CSP of MNE.

        :param n_components:
        :param reg:
        :param log:
        :param cov_est:
        :param transform_into:
        :param norm_trace:
        :param cov_method_params:
        :param rank:
        :param component_order:
        """
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components

        self.reg = reg
        self.log = log

        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        self.cov_est = cov_est

        self.transform_into = transform_into
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.rank = rank
        self.component_order = component_order
        self._classes = 0
        self.filters_ = None
        self.mean_ = 0
        self.std_ = 0

    def _calc_covariance(self, X, ddof=0):
        """
        Calculate the covariance based on numpy implementation

        :param X:
        :param ddof:ddof=1 will return the unbiased estimate, even if both fweights and aweights are specified
                    ddof=0 will return the simple average
        :return:
        """
        X -= X.mean(axis=1)[:, None]
        N = X.shape[1]
        return np.dot(X, X.T.conj()) / float(N - ddof)

    def _compute_covariance_matrices(self, X, y):
        """
        Compute covariance to every class

        :param X:ndarray, shape (n_epochs, n_channels, n_times)
                The data on which to estimate the CSP.
        :param y:array, shape (n_epochs,)
                The class for each epoch.

        :return:instance of CSP
        """
        _, n_channels, _ = X.shape
        covs = []

        for this_class in self._classes:
            x_class = X[y == this_class]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(n_channels, -1)
            # calc covar matrix for class
            covar_matrix = self._calc_covariance(x_class)
            covs.append(covar_matrix)
        return np.stack(covs)

    def fit(self, X, y):
        """
        Estimate the CSP decomposition on epochs.

        :param X:ndarray, shape (n_epochs, n_channels, n_times)
                The data on which to estimate the CSP.
        :param y:array, shape (n_epochs,)
                The class for each epoch.

        :return:instance of CSP
        """
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        else:
            covs = self._compute_covariance_matrices(X, y)
            eigen_vectors, eigen_values = self._decompose_covs(covs)
            ix = self._order_components(eigen_values)
            eigen_vectors = eigen_vectors[:, ix]
            self.filters_ = eigen_vectors.T
            pick_filters = self.filters_[:self.n_components]

            X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
            X = (X ** 2).mean(axis=2)

            # Standardize features
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)

            return self

    def transform(self, X):
        """
        Estimate epochs sources given the CSP filters.

        :param X: ndarray
        :return: ndarray
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')
        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        if self.transform_into == 'average_power':
            X = (X ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        return X

    def fit_transform(self, X, y):
        """
        Appluy fit and transform

        :param X:
        :param y:
        :param kwargs:
        :return:
        """
        self.fit(X, y)
        return self.transform(X)

    def _decompose_covs(self, covs):
        """
         Return the eigenvalues and eigenvectors of a complex Hermitian ( conjugate symmetric )

        :param covs:
        :return:
        """
        from scipy import linalg
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            raise Exception("Not Handled")
        return eigen_vectors, eigen_values

    def _order_components(self, eigen_values):
        """
        Sort components using the mutual info method.

        :param eigen_values:
        :return:
        """
        n_classes = len(self._classes)
        if n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        else:
            raise Exception("Not Handled")
        return ix

# Fichier pipeline.py
# coding: utf-8

import numpy as np
import os
import mne
import matplotlib.pyplot as plt

from CSP import CSP  # use my own CSP

from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import make_standard_montage
# from mne.decoding import CSP  # use mne CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump

data = 'mne_data'
tmin, tmax = -1., 4.
event_ids=dict(T1=0, T2=1)
subjects = [2]  # Use data of subject number 1
R1 = [6, 10, 14]  # motor imagery: hands vs feet
R2 = [4, 8, 12]  # motor imagery: left hand vs right hand

raw_fnames = os.listdir(f"{data}/{subjects[0]}")
dataset = []
subject = []
sfreq = None

for i, f in enumerate(raw_fnames):
    if f.endswith(".edf") and int(f.split('R')[1].split(".")[0]) in R2:
        subject_data = mne.io.read_raw_edf(os.path.join(f"{data}/{subjects[0]}", f), preload=True)
        if sfreq is None:
            sfreq = subject_data.info["sfreq"]
        if subject_data.info["sfreq"] == sfreq:
            subject.append(subject_data)
        else:
            break
dataset.append(mne.concatenate_raws(subject))
raw = concatenate_raws(dataset)

print(raw)
print(raw.info)
print(raw.info["ch_names"])
print(raw.annotations)

raw.rename_channels(lambda x: x.strip('.'))
montage = make_standard_montage('standard_1020')
eegbci.standardize(raw)

# create 10-05 system
raw.set_montage(montage)

# plot
montage = raw.get_montage()
p = montage.plot()
p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})

# data filtered
data_filter = raw.copy()
data_filter.set_montage(montage)
data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
p = mne.viz.plot_raw(data_filter, scalings={"eeg": 75e-6})

# get events
events, _ = events_from_annotations(data_filter, event_id=event_ids)
picks = mne.pick_types(data_filter.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
epochs = mne.Epochs(data_filter, events, event_ids, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)
labels = epochs.events[:, -1]

epochs_data_train = epochs.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)


# Assemble a classifier
scores = []
lda = LDA()
lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
svc = SVC(gamma='auto')


csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)


clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores_lda = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
mean_scores_lda, std_scores_lda = np.mean(scores_lda), np.std(scores_lda)
clf = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
scores_ldashrinkage = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores_svc = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
mean_scores_svc, std_scores_svc = np.mean(scores_svc), np.std(scores_svc)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("LDA Classification accuracy: %f / Chance level: %f" % (np.mean(scores_lda), class_balance))
print("LDA SHRINKED Classification accuracy: %f / Chance level: %f" % (np.mean(scores_ldashrinkage), class_balance))
print("SVC Classification accuracy: %f / Chance level: %f" % (np.mean(scores_svc), class_balance))

sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data_train.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv.split(epochs_data_train):
    print(1)
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda_shrinkage.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data_train[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda_shrinkage.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()

lda_shrinkage.fit(csp.fit_transform(epochs_data_train, labels), labels)
try:
    os.remove('model.joblib')
except OSError:
    pass
dump(lda_shrinkage, 'model.joblib')


# Prediction

pivot = int(0.5 * len(epochs_data_train))

clf = clf.fit(epochs_data_train[:pivot], labels[:pivot])
try :
    p = clf.named_steps["CSP"].plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
except AttributeError:
    print("Method not implemented")

print("X shape= ", epochs_data_train[pivot:].shape, "y shape= ", labels[pivot:].shape)

scores = []
for n in range(epochs_data_train[pivot:].shape[0]):
    pred = clf.predict(epochs_data_train[pivot:][n:n + 1, :, :])
    print("n=", n, "pred= ", pred, "truth= ", labels[pivot:][n:n + 1])
    scores.append(1 - np.abs(pred[0] - labels[pivot:][n:n + 1][0]))
print("Mean acc= ", np.mean(scores))
pass

# Fichier predict.py
# coding: utf-8

import numpy as np
import os
import mne

from joblib import load
from training import *

DATA_DIR = "mne_data"
PREDICT_MODEL = "final_model.joblib"
SUBJECTS = [42]


def predict():
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    # Fetch Data
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=os.listdir(f"{DATA_DIR}/{SUBJECTS[0]}"))))
    labels, epochs = fetch_events(filter_data(raw))
    epochs = epochs.get_data()

    print("X shape= ", epochs.shape, "y shape= ", labels.shape)

    scores = []
    for n in range(epochs.shape[0]):
        pred = clf.predict(epochs[n:n + 1, :, :])
        print("pred= ", pred, "truth= ", labels[n:n + 1])
        scores.append(1 - np.abs(pred[0] - labels[n:n + 1][0]))
    print("Mean acc= ", np.mean(scores))


if __name__ == "__main__":
    predict()

# Fichier training.py
# coding: utf-8

import numpy as np
import os
import mne


from CSP import CSP  # use my own CSP

from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import make_standard_montage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump

DATA_DIR = "mne_data"
SUBJECTS = [42]
RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand


def fetch_events(data_filtered, tmin=-1., tmax=4.):
    event_ids = dict(T1=0, T2=1)
    events, _ = events_from_annotations(data_filtered, event_id=event_ids)
    picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    return labels, epochs


def filter_data(raw, montage=make_standard_montage('standard_1020')):
    data_filter = raw.copy()
    data_filter.set_montage(montage)
    data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
    p = mne.viz.plot_raw(data_filter, scalings={"eeg": 75e-6})
    return data_filter


def prepare_data(raw, montage=make_standard_montage('standard_1020')):
    raw.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(raw)
    raw.set_montage(montage)

    # plot
    montage = raw.get_montage()
    p = montage.plot()
    p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
    return raw


def fetch_data(raw_fnames, sfreq=None):
    dataset = []
    subject = []
    for i, f in enumerate(raw_fnames):
        if f.endswith(".edf") and int(f.split('R')[1].split(".")[0]) in RUNS1:
            subject_data = mne.io.read_raw_edf(os.path.join(f"{DATA_DIR}/{SUBJECTS[0]}", f), preload=True)
            if sfreq is None:
                sfreq = subject_data.info["sfreq"]
            if subject_data.info["sfreq"] == sfreq:
                subject.append(subject_data)
            else:
                break
    dataset.append(mne.concatenate_raws(subject))
    raw = concatenate_raws(dataset)
    return raw


def training():
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=os.listdir(f"{DATA_DIR}/{SUBJECTS[0]}"))))
    labels, epochs = fetch_events(filter_data(raw))

    epochs_data_train = epochs.get_data()
    lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    csp = CSP()

    clf = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
    scores_ldashrinkage = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)

    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("LDA SHRINKED Classification accuracy: %f / Chance level: %f" % (np.mean(scores_ldashrinkage), class_balance))
    print(f"Mean Score Model {mean_scores_ldashrinkage}")
    print(f"Std Score Model {std_scores_ldashrinkage}")

    # save pipeline
    clf = clf.fit(epochs_data_train, labels)
    dump(clf, "final_model.joblib")
    print("model saved to final_model.joblib")
    pass


if __name__ == '__main__':
    training()

# Ficheir utils.py
# coding: utf-8

import os
import mne


def safe_opener(file):
    """
    Function used to safely open the csv file
    :param file: name of the file
    :return data: containing the training set
    """
    cwd = os.getcwd()
    try:
        f = open(os.path.join(cwd, file), 'rb')
        raw = mne.io.read_raw_edf(os.path.join(cwd, file), preload=True)
    except Exception as e:
        print("Cant open the EDF file passed as argument :" + file)
        raise e
    return raw