from typing import List
import mne
import logging
import numpy as np
import os
import pickle
from mne.datasets import eegbci
from mne import Epochs, make_fixed_length_events, pick_types, events_from_annotations
from dataclasses import asdict, dataclass
from dataclassModels import PreProcessConfiguration
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)


@dataclass
class Experiment:
    id: int
    name: str
    runs: List[int]





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
    for subject, i in zip(subjects, range(len(subjects))):
        print(f"----------Experiment {expId} - Preprocessing subject {subject} ({i+1}/{subjectsTotal})----------")
        if expId == 5:
            epochs, labels = preprocessOneSubjectEyesOpenClosed(subject, config)
        else:
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


def preprocessOneSubjectOneExperiment(subject, runs, config: PreProcessConfiguration, display=False):
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
    
    if display:
        rawBrutConcat.plot(n_channels=3, duration = 2.5)
        rawBrutConcat.plot_psd(average=True)
        rawBrutConcat.plot_psd(average=False)
    rawBrutConcat.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')
    if display:
        rawBrutConcat.plot_psd(average=True)
        rawBrutConcat.plot_psd(average=False)

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


def preprocessOneSubjectEyesOpenClosed(subject, config: PreProcessConfiguration):
    baseline_runs = [1, 2]  # Run 1: Eyes Open, Run 2: Eyes Closed
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

    return epochs, labels


