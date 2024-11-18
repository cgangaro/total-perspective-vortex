import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)})

import mne
from mne.decoding import LinearModel, Vectorizer, get_coef, Scaler, CSP, SPoC, UnsupervisedSpatialFilter
mne.set_log_level('WARNING')

import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import mne
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
import numpy as np
import mne
from newPreprocess import newPreprocess
matplotlib.use("webagg")


def main():

    subjects1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    subjects2 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    subjects3 = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    subjects = subjects1 + subjects2 + subjects3

    # raw_files = []

    # for person_number in subject:
    #     for i in run_execution:
    #         raw_execution = NewPreprocessing.loadRawFile(person_number, i)
    #         events, _ = mne.events_from_annotations(raw_execution, event_id=dict(T0=1,T1=2,T2=3))
    #         mapping = {1:'rest', 2: 'do/feet', 3: 'do/hands'}
    #         annot_from_events = mne.annotations_from_events(
    #             events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
    #             orig_time=raw_execution.info['meas_date'])
    #         raw_execution.set_annotations(annot_from_events)
    #         raw_files.append(raw_execution)

    # raw = concatenate_raws(raw_files)

    # events, event_dict = mne.events_from_annotations(raw)
    # print(raw.info)
    # print(event_dict)
    # picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    # print(picks)
    # biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    # # biosemi_montage.plot()
    # # plt.show()
    # eegbci.standardize(raw)  # set channel names
    # montage = make_standard_montage('standard_1005')
    # raw.set_montage(montage)
    # # raw.plot(n_channels=3, duration = 2.5)
    # # # plt.show()
    # # raw.plot_psd(average=True)
    # # # plt.show()
    # # raw.plot_psd(average=False)
    # # # plt.show()
    # # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
    # # fig.subplots_adjust(right=0.7)  # make room for legend
    # raw.filter(5., 40., fir_design='firwin', skip_by_annotation='edge')
    # # raw.plot_psd(average=True)
    # # plt.show()
    # # raw.plot_psd(average=False)
    # # plt.show()
    # raw_fastica = run_ica(raw, 'fastica', picks)
    # # raw.plot(n_channels=25, start=0, duration=40,scalings=dict(eeg=250e-6))
    # # raw_fastica.plot(n_channels=25, start=0, duration=40,scalings=dict(eeg=250e-6))
    # # plt.show()
    # event_id = {'do/feet': 1, 'do/hands': 2}
    # tmin, tmax = -1., 4.

    # events, event_dict = mne.events_from_annotations(raw, event_id=event_id)
    # print(event_dict)

    # epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    # print(epochs)
    # print("Epochs data shape : ", epochs.get_data().shape)
    # labels = epochs.events[:, -1] - 1
    # print(labels)

    dataPreprocessed = newPreprocess(subjects)

    for data in dataPreprocessed:
        print(data['epochs'])
        print(data['labels'])
        epochs = data['epochs']
        labels = data['labels']
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

if __name__ == "__main__":
    main()