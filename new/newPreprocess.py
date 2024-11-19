import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap, Xdawn)
from preprocessing.newPreprocessing import NewPreprocessing
from dataclasses import dataclass

@dataclass
class PreProcessConfiguration:
    withTargetSfreq: bool
    makeMontage: bool
    montageShape: str
    lowFilter: float
    highFilter: float
    with128Hz: bool
    targetSfreq: float


@dataclass
class Configuration:
    preProcess: PreProcessConfiguration


def newPreprocess(subjects, config: PreProcessConfiguration):
    print("newPreprocess")
    experiments = {
        0: [3, 7, 11],
        1: [4, 8, 12],
        2: [5, 9, 13],
        3: [6, 10, 14]
        # 4: [3, 5, 7, 9, 11, 13],
        # 5: [4, 6, 8, 10, 12, 14]
    }
    dataPreprocessed = []
    for expId in experiments:
        print(f"Preprocessing experiment {expId}")
        epochs, labels = getExperimentData(subjects, experiments[expId], config)
        dataPreprocessed.append(
            {
                'experiment': expId,
                'epochs': epochs,
                'labels': labels
            }
        )
    if False:
        epochs, labels = getExperimentData(subjects, [], config, realVsImaginary=True)
        dataPreprocessed.append(
            {
                'experiment': 4,
                'epochs': epochs,
                'labels': labels
            }
        )
    return dataPreprocessed

def getRealVsImaginaryData(subjects, imaginaryRuns, realRuns, config: Configuration):
    raw_files = []
    for person in subjects:
        for task in imaginaryRuns:
            raw_execution = NewPreprocessing.loadRawFile(person, task)
            events, _ = events_from_annotations(raw_execution, event_id=dict(T0=1,T1=2,T2=2))
            mapping = {1:'rest', 2: 'imaginary'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
                orig_time=raw_execution.info['meas_date'])
            raw_execution.set_annotations(annot_from_events)
            if config.with128Hz:
                if raw_execution.info['sfreq'] == config.targetSfreq:
                    raw_files.append(raw_execution)
            else:
                if raw_execution.info['sfreq'] != config.targetSfreq:
                    raw_execution.resample(config.targetSfreq, npad="auto")
                raw_files.append(raw_execution)
        for task in realRuns:
            raw_execution = NewPreprocessing.loadRawFile(person, task)
            events, _ = events_from_annotations(raw_execution, event_id=dict(T0=1,T1=3,T2=3))
            mapping = {1:'rest', 3: 'real'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
                orig_time=raw_execution.info['meas_date'])
            raw_execution.set_annotations(annot_from_events)
            if config.with128Hz:
                if raw_execution.info['sfreq'] == config.targetSfreq:
                    raw_files.append(raw_execution)
            else:
                if raw_execution.info['sfreq'] != config.targetSfreq:
                    raw_execution.resample(config.targetSfreq, npad="auto")
                raw_files.append(raw_execution)
    return raw_files

def getExperimentData(subjects, runs, config: Configuration, realVsImaginary=False):
    print("getExperimentData, subjects: ", subjects, "runs: ", runs)
    realRuns = [3, 7]
    imaginaryRuns = [4, 8]
    raw_files = []
    if realVsImaginary:
        print("real vs imaginary")
        raw_files = getRealVsImaginaryData(subjects, imaginaryRuns, realRuns, config)
    else:
        print("not real vs imaginary")
        for person in subjects:
            for task in runs:
                raw_execution = NewPreprocessing.loadRawFile(person, task)
                events, _ = events_from_annotations(raw_execution, event_id=dict(T0=1,T1=2,T2=3))
                mapping = {1:'rest', 2: 'do/feet', 3: 'do/hands'}
                annot_from_events = mne.annotations_from_events(
                    events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
                    orig_time=raw_execution.info['meas_date'])
                raw_execution.set_annotations(annot_from_events)
                if config.with128Hz:
                    if raw_execution.info['sfreq'] == config.targetSfreq:
                        raw_files.append(raw_execution)
                else:
                    if raw_execution.info['sfreq'] != config.targetSfreq:
                        raw_execution.resample(config.targetSfreq, npad="auto")
                    raw_files.append(raw_execution)
    # if config.withTargetSfreq:
    #     target_sfreq = 160
    #     for i in range(len(raw_files)):
    #         if raw_files[i].info['sfreq'] != target_sfreq:
    #             raw_files[i].resample(target_sfreq, npad="auto")
    raw = concatenate_raws(raw_files)

    events, event_dict = events_from_annotations(raw)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    eegbci.standardize(raw)
    if config.makeMontage:
        montage = make_standard_montage(config.montageShape)
        raw.set_montage(montage)
    raw.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')
    if realVsImaginary:
        event_id = {'imaginary': 1, 'real': 2}
    else:
        event_id = {'do/feet': 1, 'do/hands': 2}
    tmin, tmax = -1., 3.
    # raw = run_ica(raw, 'fastica', picks)
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id)
    print(event_dict)
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    print(epochs)
    print("Epochs data shape : ", epochs.get_data().shape)
    labels = epochs.events[:, -1] - 1
    print(labels)
    return epochs, labels

def run_ica(raw, method, picks, fit_params=None):
    raw_corrected = raw.copy()
    n_components=10
    print('test')
    ica = ICA(n_components=n_components, method=method, fit_params=fit_params, random_state=97)
    print('test2')
    ica.fit(raw_corrected, picks=picks)
    print('test3')
    eog_indices, scores = ica.find_bads_eog(raw, ch_name='Fpz',threshold=1.5)
    print('test4')
    ica.exclude.extend(eog_indices)
    print('test5')
    raw_corrected = ica.apply(raw_corrected, n_pca_components = n_components, exclude = ica.exclude)
    print('test6')
    print(ica.exclude, ica.labels_)
    return raw_corrected
