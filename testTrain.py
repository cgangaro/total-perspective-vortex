import numpy as np
import matplotlib.pyplot as plt
import warnings
from time import time
import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)})

import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap, Xdawn)
from mne.time_frequency import AverageTFR
from mne.channels import make_standard_montage
from mne.decoding import LinearModel, Vectorizer, get_coef, Scaler, CSP, SPoC, UnsupervisedSpatialFilter
mne.set_log_level('WARNING')

import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
import mne
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from cspTransformer import CSPTransformer
from preProcess import preProcess, Configuration, PreProcessConfiguration, splitData, newSplitData
from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import mne
import os
from mne import pick_types
from preprocessing.preprocessing import Preprocessing
from preprocessing.newPreprocessing import NewPreprocessing
from preprocessing.featureExtraction import FeatureExtraction
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
matplotlib.use("webagg")


def main():

    subject = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 1, 4
    run_execution = [5, 9, 13]
    run_imagery = [6, 10, 14]

    raw_files = []

    for person_number in subject:
        for i in run_execution:
            # raw_files_execution = [read_raw_edf(f, preload=True, stim_channel='auto') for f in eegbci.load_data(person_number, i)]
            # raw_execution = concatenate_raws(raw_files_execution)
            raw_execution = NewPreprocessing.loadRawFile(person_number, i)
            events, _ = mne.events_from_annotations(raw_execution, event_id=dict(T0=1,T1=2,T2=3))
            mapping = {1:'rest', 2: 'do/feet', 3: 'do/hands'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
                orig_time=raw_execution.info['meas_date'])
            raw_execution.set_annotations(annot_from_events)
            raw_files.append(raw_execution)

    raw = concatenate_raws(raw_files)

    events, event_dict = mne.events_from_annotations(raw)
    print(raw.info)
    print(event_dict)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    print(picks)
    biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    # biosemi_montage.plot()
    # plt.show()
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    # raw.plot(n_channels=3, duration = 2.5)
    # # plt.show()
    # raw.plot_psd(average=True)
    # # plt.show()
    # raw.plot_psd(average=False)
    # # plt.show()
    # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
    # fig.subplots_adjust(right=0.7)  # make room for legend
    raw.filter(5., 40., fir_design='firwin', skip_by_annotation='edge')
    # raw.plot_psd(average=True)
    # plt.show()
    # raw.plot_psd(average=False)
    # plt.show()
    raw_fastica = run_ica(raw, 'fastica', picks)
    # raw.plot(n_channels=25, start=0, duration=40,scalings=dict(eeg=250e-6))
    # raw_fastica.plot(n_channels=25, start=0, duration=40,scalings=dict(eeg=250e-6))
    # plt.show()
    event_id = {'do/feet': 1, 'do/hands': 2}
    tmin, tmax = -1., 4.

    events, event_dict = mne.events_from_annotations(raw, event_id=event_id)
    print(event_dict)

    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    print(epochs)
    print("Epochs data shape : ", epochs.get_data().shape)
    labels = epochs.events[:, -1] - 1
    print(labels)

    score_pipeline = -1
    models_pipeline = None

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    cv = ShuffleSplit(10, test_size=0.4, random_state=42)

    # # Assemble a classifier
    csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
    rfc = RandomForestClassifier(n_estimators=150, random_state=42)
    print(f"Avant pipeline : epochs_data type: {epochs_data.dtype}, shape: {epochs_data.shape}")
    print(f"Labels type: {labels.dtype}, shape: {labels.shape}")
    # Use scikit-learn Pipeline with cross_val_score function
    clf = make_pipeline(csp, rfc)
    scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

    # Printing the results
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    if scores.mean() > score_pipeline:
        score_pipeline = scores.mean()
        models_pipeline = clf

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)
    # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    # plt.show()

    return


def run_ica(raw, method, picks, fit_params=None):
    raw_corrected = raw.copy()
    n_components=20
    
    ica = ICA(n_components=n_components, method=method, fit_params=fit_params, random_state=97)
    t0 = time()
    ica.fit(raw_corrected, picks=picks)
    fit_time = time() - t0
    title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
    # ica.plot_components(title=title)
    # plt.show()
    
    eog_indices, scores = ica.find_bads_eog(raw, ch_name='Fpz',threshold=1.5)
    # ica.plot_scores(scores, exclude=eog_indices)  # look at r scores of components
    ica.exclude.extend(eog_indices) 
    raw_corrected = ica.apply(raw_corrected, n_pca_components = n_components, exclude = ica.exclude)
    print(ica.exclude, ica.labels_)
    return raw_corrected


if __name__ == "__main__":
    main()