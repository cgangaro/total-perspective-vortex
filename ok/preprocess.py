import mne
import logging
import numpy as np
import os
import pickle
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from dataclasses import dataclass
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)

@dataclass
class PreProcessConfiguration:
    dataLocation: str = "/home/cgangaro/sgoinfre/mne_data"
    loadData: bool = False
    saveData: bool = True
    makeMontage: bool = True
    montageShape: str = "standard_1020"
    resample: bool = True
    resampleFreq: float = 90.0
    lowFilter: float = 8.0
    highFilter: float = 36.0
    ica: bool = True
    icaComponents: int = 20
    eog: bool = True
    epochsTmin: float = -1.0
    epochsTmax: float = 3.0


def preprocess(subjects, experiments, config: PreProcessConfiguration, saveDirectory):
    os.makedirs(saveDirectory, exist_ok=True)
    dataPreprocessed = []
    for expId in experiments:
        if not config.loadData:
            print(f"Preprocessing experiment {expId}")
            epochs, labels, subject_ids = preprocessOneExperiment(subjects, experiments[expId], config)
            if config.saveData:
                filename = os.path.join(saveDirectory, f"experiment_{expId}.pkl")
                save_preprocessed_data({'epochs': epochs, 'labels': labels, 'subject_ids': subject_ids}, filename)
            dataPreprocessed.append(
                {
                    'experiment': expId,
                    'epochs': epochs,
                    'labels': labels,
                    'subject_ids': subject_ids
                }
            )
        else:
            filename = os.path.join(saveDirectory, f"experiment_{expId}.pkl")
            data = load_preprocessed_data(filename)
            dataPreprocessed.append(
                {
                    'experiment': expId,
                    'epochs': data['epochs'],
                    'labels': data['labels'],
                    'subject_ids': data['subject_ids']
                }
            )
    return dataPreprocessed

def preprocessOneExperiment(subjects, runs, config: PreProcessConfiguration):
    print(f"\n----------Preprocessing experiment for subjects: {subjects}----------\n")
    all_epochs = []
    all_labels = []
    all_subject_ids = []
    subjectsTotal = len(subjects)
    for subject, i in zip(subjects, range(len(subjects))):
        print(f"----------Preprocessing subject {subject} ({i+1}/{subjectsTotal})----------")
        epochs, labels = preprocessOneSubjectOneExperiment(subject, runs, config)
        all_epochs.append(epochs)
        all_labels.append(labels)
        subject_ids = np.full(len(labels), subject)
        all_subject_ids.append(subject_ids)
    
    epochs = mne.concatenate_epochs(all_epochs)
    labels = np.concatenate(all_labels)
    subject_ids = np.concatenate(all_subject_ids)
    print(f"\n----------Finished preprocessing experiment for subjects: {subjects}----------\n")
    return epochs, labels, subject_ids


def preprocessOneSubjectOneExperiment(subject, runs, config: PreProcessConfiguration):
    rawFnames = eegbci.load_data(subject, runs=runs, verbose="ERROR", path=config.dataLocation)
    rawBrut = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in rawFnames]
    rawBrutConcat = mne.concatenate_raws(rawBrut, verbose="ERROR")

    rawBrutConcat.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(rawBrutConcat)

    if config.makeMontage:
        rawBrutConcat.set_montage(config.montageShape)
    
    rawBrutConcat.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')

    picks = pick_types(rawBrutConcat.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    if config.ica:
        ica = mne.preprocessing.ICA(n_components=config.icaComponents)
        ica = mne.preprocessing.ICA(n_components=config.icaComponents, random_state=97, max_iter=800)
        ica.fit(rawBrutConcat, picks=picks)

        if config.eog:
            channels = ['Fp1', 'Fp2', 'Fpz', 'AFz', 'AF3', 'AF4']
            canal_eog = []
            for canal in channels:
                if canal in rawBrutConcat.ch_names:
                    canal_eog.append(canal)
            # print(f"-------------------Canaux EOG trouvés: {canal_eog}-------------------")
            if canal_eog != []:
                # print(f"Utilisation du canal {canal_eog} pour la détection des artéfacts EOG.")
                eog_indices, eog_scores = ica.find_bads_eog(rawBrutConcat, ch_name=canal_eog)
                ica.exclude = eog_indices
                rawBrutConcat = ica.apply(rawBrutConcat)
            else:
                print("Aucun canal frontal approprié trouvé pour la détection des artéfacts EOG. La suppression des artéfacts EOG est ignorée pour ce sujet.")

    if config.resample:
        rawBrutConcat.resample(config.resampleFreq, npad="auto")

    eventsId = dict(T1=0,T2=1)
    events, _ = events_from_annotations(rawBrutConcat, event_id=eventsId)

    epochs = Epochs(rawBrutConcat, events, eventsId, config.epochsTmin, config.epochsTmax, proj=True, picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]

    return epochs, labels

def save_preprocessed_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def load_preprocessed_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {filename}")
    return data