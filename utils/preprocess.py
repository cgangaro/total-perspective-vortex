from typing import List
import mne
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from mne.datasets import eegbci
from mne import Epochs, make_fixed_length_events, pick_types, events_from_annotations
from utils.dataclassModels import PreProcessConfiguration
from sklearn.model_selection import train_test_split

from utils.preprocessTest2 import pre_process_data
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)


def preprocess(saveDirectory, config: PreProcessConfiguration, subjects = [], experiments = [], saveData = False, loadData = False):
    if saveData:
        os.makedirs(saveDirectory, exist_ok=True)
    dataPreprocessed = []
    print(f"DataDirectory: {saveDirectory}, experiments: {experiments}, loadData: {loadData}, saveData: {saveData}")
    for exp in experiments:
        if not loadData:
            print(f"Preprocessing experiment {exp.id} - {exp.name}")
            epochs, labels, subject_ids = preprocessOneExperiment(subjects, exp.runs, exp.id, config)
            if saveData:
                filename = os.path.join(saveDirectory, f"experiment_{exp.id}.pkl")
                save_preprocessed_data({'epochs': epochs, 'labels': labels, 'subject_ids': subject_ids}, filename)
            dataPreprocessed.append(
                {
                    'experiment': exp.id,
                    'experiment_name': exp.name,
                    'epochs': epochs,
                    'labels': labels,
                    'subject_ids': subject_ids
                }
            )
        else:
            filename = os.path.join(saveDirectory, f"experiment_{exp.id}.pkl")
            data = load_preprocessed_data(filename)
            dataPreprocessed.append(
                {
                    'experiment': exp.id,
                    'experiment_name': exp.name,
                    'epochs': data['epochs'],
                    'labels': data['labels'],
                    'subject_ids': data['subject_ids']
                }
            )
    return dataPreprocessed


def preprocessMEGA(saveDirectory, config: PreProcessConfiguration, subjects=[], experiments=[], saveData=False, loadData=False):
    if saveData:
        os.makedirs(saveDirectory, exist_ok=True)
    
    dataTrainPreprocessed = []
    dataTestPreprocessed = []
    
    print(f"DataDirectory: {saveDirectory}, experiments: {experiments}, loadData: {loadData}, saveData: {saveData}")
    
    for exp in experiments:
        if not loadData:
            print(f"Preprocessing experiment {exp.id} - {exp.name}")
            train_data, test_data = preprocessOneExperimentMEGA(subjects, exp.runs, exp.id, config)
            
            if saveData:
                train_filename = os.path.join(saveDirectory, f"experiment_{exp.id}_train.pkl")
                test_filename = os.path.join(saveDirectory, f"experiment_{exp.id}_test.pkl")
                save_preprocessed_data(train_data, train_filename)
                save_preprocessed_data(test_data, test_filename)
            print(f"Train data: {train_data}")
            print(f"dataTrainPreprocessed: {dataTrainPreprocessed}")
            dataTrainPreprocessed.append(train_data)
            print(f"dataTrainPreprocessed: {dataTrainPreprocessed}")
            dataTestPreprocessed.append(test_data)
        else:
            train_filename = os.path.join(saveDirectory, f"experiment_{exp.id}_train.pkl")
            test_filename = os.path.join(saveDirectory, f"experiment_{exp.id}_test.pkl")
            train_data = load_preprocessed_data(train_filename)
            test_data = load_preprocessed_data(test_filename)
            dataTrainPreprocessed.append(train_data)
            dataTestPreprocessed.append(test_data)
    
    return dataTrainPreprocessed, dataTestPreprocessed

def preprocessOneExperimentMEGA(subjects, runs, expId, config: PreProcessConfiguration):
    print(f"\n----------Preprocessing experiment for subjects: {subjects}----------\n")
    subjectsTotal = len(subjects)
    ica = None
    train_epochs_all = []
    train_labels_all = []
    test_epochs_all = []
    test_labels_all = []

    for subject, i in zip(subjects, range(len(subjects))):
        print(f"----------Experiment {expId} - Preprocessing subject {subject} ({i+1}/{subjectsTotal})----------")
        
        if runs == [1, 2]:
            epochs, labels, ica = preprocessOneSubjectEyesOpenClosed(subject, config, ica)
        elif expId == 10:
            epochs = pre_process_data(subject, ['do/hands', 'do/feet'])
        elif expId == 11:
            epochs = pre_process_data(subject, ['imagine/hands', 'imagine/feet'])
        else:
            epochs, labels, event_id = preprocessOneSubjectOneExperiment(subject, runs, config, ica)
        
        # Split epochs into train and test sets
        X = epochs.get_data()
        y = epochs.events[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Subject {subject}: {len(X_train)} train epochs, {len(X_test)} test epochs, unique labels: {np.unique(y)}, x shape: {X.shape}, y shape: {y.shape}")
        
        train_epochs_all.append(X_train)
        train_labels_all.extend(y_train)
        test_epochs_all.append(X_test)
        test_labels_all.extend(y_test)

    trainEpochs = np.concatenate(train_epochs_all, axis=0)
    trainLabels = np.array(train_labels_all)
    testEpochs = np.concatenate(test_epochs_all, axis=0)
    testLabels = np.array(test_labels_all)
    print(f"\n----------Finished preprocessing experiment for subjects: {subjects}----------\n")

    train_data = {
        'experiment': expId,
        'epochs': trainEpochs,
        'labels': trainLabels,
        'event_id': event_id
    }
    test_data = {
        'experiment': expId,
        'epochs': testEpochs,
        'labels': testLabels,
        'event_id': event_id
    }

    return train_data, test_data


def preprocessOneExperiment(subjects, runs, expId, config: PreProcessConfiguration):
    print(f"\n----------Preprocessing experiment for subjects: {subjects}----------\n")
    all_epochs = []
    all_labels = []
    all_subject_ids = []
    subjectsTotal = len(subjects)
    ica = None
    for subject, i in zip(subjects, range(len(subjects))):
        print(f"----------Experiment {expId} - Preprocessing subject {subject} ({i+1}/{subjectsTotal})----------")
        if runs == [1, 2]:
            epochs, labels, ica = preprocessOneSubjectEyesOpenClosed(subject, config, ica)
        elif expId == 10:
            [epochs, labels, _] = pre_process_data(subject, ['do/hands', 'do/feet'])
        elif expId == 11:
            [epochs, labels, _] = pre_process_data(subject, ['imagine/hands', 'imagine/feet'])
        else:
            epochs, labels, ica = preprocessOneSubjectOneExperiment(subject, runs, config, ica)
        all_epochs.append(epochs)
        all_labels.append(labels)
        subject_ids = np.full(len(labels), subject)
        all_subject_ids.append(subject_ids)
    
    epochs = mne.concatenate_epochs(all_epochs)
    labels = np.concatenate(all_labels)
    subject_ids = np.concatenate(all_subject_ids)
    print(f"\n----------Finished preprocessing experiment for subjects: {subjects}----------\n")
    return epochs, labels, subject_ids


def preprocessOneSubjectOneExperiment(subject, runs, config: PreProcessConfiguration, ica, display=False):
    print(f"Preprocessing subject {subject} for runs {runs}")
    rawFnames = eegbci.load_data(subject, runs=runs, verbose="ERROR", path=config.dataLocation)
    rawBrut = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in rawFnames]
    rawBrutConcat = mne.concatenate_raws(rawBrut, verbose="ERROR")

    rawBrutConcat.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(rawBrutConcat)

    if config.makeMontage:
        rawBrutConcat.set_montage(config.montageShape)
        if display:
            biosemi_montage = mne.channels.make_standard_montage(config.montageShape)
            biosemi_montage.plot()
            plt.show()
    
    if display:
        rawBrutConcat.plot(n_channels=3, duration = 2.5)
        plt.show()
        rawBrutConcat.plot_psd(average=True)
        plt.show()
        rawBrutConcat.plot_psd(average=False)
        plt.show()
    rawBrutConcat.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')
    if display:
        rawBrutConcat.plot_psd(average=True)
        plt.show()
        rawBrutConcat.plot_psd(average=False)
        plt.show()

    picks = pick_types(rawBrutConcat.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    if config.ica:
        if ica is None:
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

    # good_channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]Cz, C1, C2, C3, C4, FC3, FC4, CP3, CP4
    good_channels = ["Cz", "C1", "C2", "C3", "C4", "FC3", "FC4", "CP3", "CP4"]
    channels = rawBrutConcat.info["ch_names"]
    print(f"Channels: {channels}")
    bad_channels = [x for x in channels if x not in good_channels]
    print(f"Bad channels: {bad_channels}")
    rawBrutConcat.drop_channels(bad_channels)
    print(f"Channels after dropping: {rawBrutConcat.info['ch_names']}")
    eventsId = dict(T1=1,T2=2)
    events, event_id = events_from_annotations(rawBrutConcat, event_id=eventsId)

    picks = pick_types(rawBrutConcat.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = Epochs(rawBrutConcat, events, eventsId, config.epochsTmin, config.epochsTmax, proj=True, picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    # epochs, labels = average_over_epochs(
    #     epochs,
    #     labels,
    #     event_id
    # )
    return epochs, labels, event_id


def preprocessOneSubjectEyesOpenClosed(subject, config: PreProcessConfiguration, ica):
    baseline_runs = [1, 2]
    print(f"Preprocessing subject {subject} for runs {baseline_runs} - Eyes Open and Eyes Closed")
    raw_fnames = eegbci.load_data(subject, runs=baseline_runs, verbose="ERROR")
    all_epochs = []
    all_labels = []
    for fname, run in zip(raw_fnames, baseline_runs):
        raw = mne.io.read_raw_edf(fname, preload=True, stim_channel='auto')
        raw.rename_channels(lambda x: x.strip('.'))
        eegbci.standardize(raw)
        if config.makeMontage:
            raw.set_montage(config.montageShape)
        
        # Filtrage
        raw.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')

        # Sélection des canaux EEG
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

        # ICA pour éliminer les artefacts
        if config.ica:
            if ica is None:
                ica = mne.preprocessing.ICA(n_components=config.icaComponents, random_state=97, max_iter=800)
            ica.fit(raw, picks=picks)

            if config.eog:
                eog_channels = ['Fp1', 'Fp2', 'Fpz', 'AFz', 'AF3', 'AF4']
                eog_chs_present = [ch for ch in eog_channels if ch in raw.ch_names]
                if eog_chs_present:
                    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_chs_present)
                    ica.exclude = eog_indices
                    raw = ica.apply(raw)
                else:
                    print("No frontal channels found for EOG artifact detection. Skipping EOG artifact removal.")

        # Rééchantillonnage
        if config.resample:
            raw.resample(config.resampleFreq, npad="auto")

        # Détermination de la description en fonction du run
        if run == 1:
            description = 'Eyes Open'
            event_id = {'Eyes Open': 0}
        elif run == 2:
            description = 'Eyes Closed'
            event_id = {'Eyes Closed': 1}
        else:
            description = 'Unknown'

        # Création des événements à intervalles réguliers
        epoch_duration = 1.0  # Durée de chaque epoch en secondes
        overlap = 0.0         # Chevauchement entre les epochs en secondes
        events = make_fixed_length_events(raw, start=0, stop=None, duration=epoch_duration - overlap, overlap=overlap, first_samp=True, id=0)

        # Mise à jour des identifiants d'événements en fonction de la condition
        events[:, 2] = 0 if description == 'Eyes Open' else 1

        # Création des epochs
        tmin = 0.0
        tmax = epoch_duration
        # good_channels = ["Cz", "C1", "C2", "C3", "C4", "FC3", "FC4", "CP3", "CP4"]O1, Oz, O2, PO7, PO3, POz, PO4, et PO8 Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8
        # good_channels = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]
        # channels = raw.info["ch_names"]
        # print(f"Channels: {channels}")
        # bad_channels = [x for x in channels if x not in good_channels]
        # print(f"Bad channels: {bad_channels}")
        # raw.drop_channels(bad_channels)
        # print(f"Channels after dropping: {raw.info['ch_names']}")
        # picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

        # Filtrer les epochs pour la condition actuelle
        epochs = epochs[description]

        labels = epochs.events[:, -1]
        print(f"Subject {subject}, Run {run}: {len(epochs)} epochs, unique labels: {np.unique(labels)}")

        all_epochs.append(epochs)
        all_labels.extend(labels)

    # Concaténer les epochs et les labels des deux runs
    epochs = mne.concatenate_epochs(all_epochs)
    labels = np.array(all_labels)

    return epochs, labels, ica


def save_preprocessed_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")


def load_preprocessed_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {filename}")
    return data

def average_over_epochs(X, y, event_id):
    # E_test, y_test, E_train, y_train = split_epochs_train_test(X, y)
    new_x = []
    new_y = []

    keys = list(event_id.keys())

    if len(X[keys[0]]) > len(X[keys[1]]):
        max_len = len(X[keys[1]])
    else:
        max_len = len(X[keys[0]])

    max_avg_size = 30
    min_amount_of_epochs = 5
    if max_len < min_amount_of_epochs * max_avg_size:
        max_avg_size = math.floor(max_len / min_amount_of_epochs)
    # Optional: averaging over multiple sizes to increase dataset size
    sizes = [max_avg_size]

    for avg_size in sizes:
        print("Averaging epochs over size: ", avg_size, "...")
        i = 0
        while i < max_len:
            x_averaged = X[keys[0]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[0]])

            x_averaged = X[keys[1]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[1]])

            if i + avg_size >= len(X):
                avg_size = len(X) - i
            i = i + avg_size

    return np.array(new_x), np.array(new_y)