from typing import List
import mne
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from mne.datasets import eegbci
from mne import Epochs, make_fixed_length_events, pick_types, events_from_annotations
from utils.dataclassModels import PreProcessConfiguration
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)


def oldPreprocess(subjects, experiments, config: PreProcessConfiguration, saveDirectory):
    os.makedirs(saveDirectory, exist_ok=True)
    dataPreprocessed = []
    for exp in experiments:
        if not config.loadData:
            print(f"Preprocessing experiment {exp.id} - {exp.name}")
            epochs, labels, subject_ids = preprocessOneExperiment(subjects, exp.runs, exp.id, config)
            if config.saveData:
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

def preprocessOneExperiment(subjects, runs, expId, config: PreProcessConfiguration):
    print(f"\n----------Preprocessing experiment for subjects: {subjects}----------\n")
    all_epochs = []
    all_labels = []
    all_subject_ids = []
    subjectsTotal = len(subjects)
    ica = None
    for subject, i in zip(subjects, range(len(subjects))):
        print(f"----------Experiment {expId} - Preprocessing subject {subject} ({i+1}/{subjectsTotal})----------")
        if expId == 5:
            epochs, labels, ica = preprocessOneSubjectEyesOpenClosed(subject, config, ica)
        elif expId == 6:
            epochs, labels = preprocessRestVsMovement(subject, config)
        elif expId == 7:
            epochs, labels = preprocessOneSubjectRealOrImaginary(subject, config)
        elif expId == 8:
            epochs, labels = preprocessOneSubjectOneExperimentAAA(subject, config)
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

    eventsId = dict(T1=0,T2=1)
    events, _ = events_from_annotations(rawBrutConcat, event_id=eventsId)

    epochs = Epochs(rawBrutConcat, events, eventsId, config.epochsTmin, config.epochsTmax, proj=True, picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
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

def preprocessOneSubjectRealOrImaginary(subject, config: PreProcessConfiguration):
    real_runs = [3, 7, 11]
    imaginary_runs = [5, 9, 13]
    all_runs = real_runs + imaginary_runs
    print(f"Preprocessing subject {subject} for runs {all_runs} - Real or Imaginary")
    raw_fnames = eegbci.load_data(subject, runs=all_runs, verbose="ERROR")
    all_epochs = []
    all_labels = []
    for fname, run in zip(raw_fnames, all_runs):
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
        if run in real_runs:
            description = 'Real'
            event_id = {'Real': 0}
            eventsId = dict(T1=0,T2=0)
        elif run in imaginary_runs:
            description = 'Imaginary'
            event_id = {'Imaginary': 1}
            eventsId = dict(T1=1,T2=1)
        else:
            description = 'Unknown'

        events, _ = events_from_annotations(raw, event_id=eventsId)
        # # Création des événements à intervalles réguliers
        # epoch_duration = 1.0  # Durée de chaque epoch en secondes
        # overlap = 0.0         # Chevauchement entre les epochs en secondes
        # events = make_fixed_length_events(raw, start=0, stop=None, duration=epoch_duration - overlap, overlap=overlap, first_samp=True, id=0)

        # # Mise à jour des identifiants d'événements en fonction de la condition
        # events[:, 2] = 0 if description == 'Real' else 1

        
        # Création des epochs
        epochs = Epochs(raw, events, event_id, config.epochsTmin, config.epochsTmax, proj=True, picks=picks, baseline=None, preload=True)

        # Filtrer les epochs pour la condition actuelle
        epochs = epochs[description]

        labels = epochs.events[:, -1]
        print(f"Subject {subject}, Run {run}: {len(epochs)} epochs, unique labels: {np.unique(labels)}")

        all_epochs.append(epochs)
        all_labels.extend(labels)

    # Concaténer les epochs et les labels des deux runs
    epochs = mne.concatenate_epochs(all_epochs)
    labels = np.array(all_labels)

    return epochs, labels


def preprocessRestVsMovement(subject, config: PreProcessConfiguration):
    """
    Prétraitement pour classifier repos (T0) vs mouvement (T1/T2) en utilisant les bandes alpha et beta.
    """
    runs = [3, 4, 5, 6, 7, 8, 9, 10]  # Runs liés aux mouvements
    print(f"Preprocessing subject {subject} for runs {runs} - Rest vs Movement")
    
    # Charger les données brutes
    raw_fnames = eegbci.load_data(subject, runs=runs, verbose="ERROR")
    raw_list = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
    raw = mne.concatenate_raws(raw_list)
    raw.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(raw)
    
    # Montage
    if config.makeMontage:
        raw.set_montage(config.montageShape)
    
    # Filtrage passe-bande pour les bandes alpha et beta
    raw_alpha = raw.copy().filter(8, 13, fir_design='firwin', skip_by_annotation='edge')  # Bande alpha
    raw_beta = raw.copy().filter(13, 30, fir_design='firwin', skip_by_annotation='edge')  # Bande beta
    
    # Sélection des canaux EEG
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
    # ICA pour éliminer les artéfacts
    if config.ica:
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
    
    # Extraction des événements
    event_id = {'T0': 0, 'T1': 1, 'T2': 1}  # T1 et T2 regroupés comme mouvement
    events, _ = events_from_annotations(raw, event_id=event_id)
    
    # Création des epochs pour les bandes alpha et beta
    # epochs_alpha = Epochs(raw_alpha, events, event_id=event_id, tmin=-0.5, tmax=2.,
    #                       proj=True, picks=picks, baseline=None, preload=True)
    epochs_beta = Epochs(raw_beta, events, event_id=event_id, tmin=-0.5, tmax=2.,
                         proj=True, picks=picks, baseline=None, preload=True)
    
    # Combiner les bandes alpha et beta en tant que caractéristiques
    # epochs_data_alpha = epochs_alpha.get_data()
    # epochs_data_beta = epochs_beta.get_data()
    # combined_epochs_data = np.concatenate([epochs_data_alpha, epochs_data_beta], axis=1)
    
    # Récupération des étiquettes
    labels = epochs_beta.events[:, -1]
    
    print(f"Subject {subject}: {len(epochs_beta)} epochs, labels: {len(labels)}, unique labels: {np.unique(labels)}")
    
    return epochs_beta, labels

def preprocessOneSubjectOneExperimentAAA(subject, config: PreProcessConfiguration):
    runs = [3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Preprocessing AAA subject {subject} for runs {runs}")
    rawFnames = eegbci.load_data(subject, runs=runs, verbose="ERROR", path=config.dataLocation)
    rawBrut = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in rawFnames]
    rawBrutConcat = mne.concatenate_raws(rawBrut, verbose="ERROR")

    rawBrutConcat.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(rawBrutConcat)

    if config.makeMontage:
        rawBrutConcat.set_montage(config.montageShape)
    
    # rawBrutConcat.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')
    # raw_alpha = raw.copy().filter(8, 13, fir_design='firwin', skip_by_annotation='edge')  # Bande alpha
    rawBrutConcat.filter(8, 13, fir_design='firwin', skip_by_annotation='edge')

    picks = pick_types(rawBrutConcat.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    if config.resample:
        rawBrutConcat.resample(config.resampleFreq, npad="auto")

    eventsId = dict(T1=0,T2=1)
    events, _ = events_from_annotations(rawBrutConcat, event_id=eventsId)

    epochs = Epochs(rawBrutConcat, events, eventsId, config.epochsTmin, config.epochsTmax, proj=True, picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    return epochs, labels