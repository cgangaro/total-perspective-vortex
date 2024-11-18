import numpy as np
import mne
import os
from mne import pick_types
from preprocessing.preprocessing import Preprocessing
from preprocessing.newPreprocessing import NewPreprocessing
from preprocessing.featureExtraction import FeatureExtraction
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class PreProcessConfiguration:
    ICA: bool
    EOG: bool
    montageKind: str
    lowCutoff: float
    highCutoff: float
    notchFreq: int
    nIcaComponents: int
    useRestMoment: bool


@dataclass
class Configuration:
    preProcess: PreProcessConfiguration
    CSPTransformerNComponents: int
    dataDir: str
    saveToFile: bool
    loadFromFile: bool


def preProcess(subjects, experiments, config: Configuration, save_to_file=False, output_dir='data', name='default'):
    if config.loadFromFile:
        return Preprocessing.loadPreprocessedData(
            subjects,
            experiments,
            data_dir=config.dataDir
        )
    data = {}
    for exp_id, runs in experiments.items():
        X_list = []
        labels_list = []
        print(f"Traitement de l'expérience {exp_id} avec les runs {runs}")
        if len(runs) == 0:
            print(f"Pas de runs pour l'expérience {exp_id}")
            data[exp_id] = {'X': np.array([]), 'labels': np.array([])}
            continue

        for subject in subjects:
            print(f"  Sujet {subject}")
            try:
                raw = NewPreprocessing.load_all_data([subject], runs)
            except Exception as e:
                print(f"Impossible de charger les données pour le sujet {subject} et les runs {runs}: {e}")
                continue

            raw = Preprocessing.makeMontage(
                raw,
                montageType=config.preProcess.montageKind,
                display=False
            )

            rawFiltered = Preprocessing.filterRawData(
                raw,
                low_cutoff=config.preProcess.lowCutoff,
                high_cutoff=config.preProcess.highCutoff,
                notch_freq=config.preProcess.notchFreq
            )

            current_sfreq = rawFiltered.info['sfreq']
            desired_sfreq = 160.0  # Fréquence d'échantillonnage cible
            if current_sfreq != desired_sfreq:
                print(f"Rééchantillonnage des données de {current_sfreq} Hz à {desired_sfreq} Hz pour le sujet {subject}")
                rawFiltered.resample(desired_sfreq, npad="auto")

            # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_id)

            picks = pick_types(rawFiltered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            # pick_types permet de recuperer les canaux de notre analyse. Il permet de filtrer les canaux que l'on veut garder.
            # raw.info contient les informations sur les canaux de notre enregistrement, comme leur nom, leur type (EEG, EOG etc...),
            # leur frequence d'echantillonnage, etc...
            # meg permet de selectionner les canaux MEG (Magnétoencéphalographie), qui correspondent à des capteurs de champs magnétiques.
            # eeg permet de selectionner les canaux EEG, qui correspondent aux electrodes placées sur le cuir chevelu.
            # stim permet de selectionner les canaux de stimulation, qui sont utilises pour marquer les evenements.
            # Les evenements peuvent etre des stimuli visuels ou auditifs par exemple.
            # eog permet de selectionner les canaux EOG, qui permettent de detecter les mouvements oculaires.
            # exclude permet de selectionner les canaux à exclure, ici on exclut les canaux marqués comme mauvais.
            if config.preProcess.ICA:
                rawFiltered = FeatureExtraction.ICAProcess(
                    rawFiltered,
                    picks,
                    config.preProcess.nIcaComponents,
                    config.preProcess.EOG
                )

            # EPOCHS
            events, event_id = mne.events_from_annotations(rawFiltered)
            print("event_id: ", event_id)
            tmin, tmax = -1., 4.
            epochs = mne.Epochs(
                rawFiltered,
                events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                proj=True,
                picks=picks,
                baseline=None,
                preload=True
            )
            labels = epochs.events[:, -1]
            # print(f"Unique labels: {np.unique(labels)}")
            # print(f"Epochs shape: {epochs.get_data().shape}")
            # print(f"Labels shape: {labels.shape}")
            epochs = epochs[['T1', 'T2']]
            # print(f"Epochs shape after selection: {epochs.get_data().shape}")
            labels = epochs.events[:, -1]
            # print(f"Unique labels after selection: {np.unique(labels)}")
            # print(f"Labels shape after selection: {labels.shape}")

            # if config.preProcess.useRestMoment:
            #     epochs = epochs[['T0', 'T1', 'T2']]
            #     labels = epochs.events[:, -1] - 1
            # else:
            #     epochs = epochs[['T1', 'T2']]
            #     labels = epochs.events[:, -1] - 2

            X = epochs.get_data()
            if len(labels) == 0 or len(X) == 0:
                print(f"Pas d'epochs pour le sujet {subject} dans l'expérience {exp_id}")
                continue

            X_list.append(X)
            labels_list.append(labels)

        X_exp = np.concatenate(X_list)
        labels_exp = np.concatenate(labels_list)

        data[exp_id] = {'X': X_exp, 'labels': labels_exp}
        print(f"Données prétraitées pour l'expérience {exp_id} : {X_exp.shape[0]} échantillons")

        if save_to_file:
            os.makedirs(output_dir, exist_ok=True)
            nbSubjects = len(subjects)
            output_path = os.path.join(output_dir, f"experiment_{exp_id}_n{nbSubjects}.npz")
            np.savez(output_path, data[exp_id])
            print(f"Données enregistrées pour l'expérience {exp_id} dans {output_path}")

    return data

    

def splitData(data, experiments):
    dataSplit = {}
    for expId in experiments:
        expData = data[expId]
        X = expData['X']
        labels = expData['labels']
        print(f"Expérience {expId} - X shape: {X.shape}, labels shape: {labels.shape} - T1 count: {np.sum(labels == 0)}, T2 count: {np.sum(labels == 1)}")
        if X.shape[0] == 0 or labels.shape[0] == 0:
            print(f"Expérience {expId} - Pas de données")
            dataSplit[expId] = {
                'X_train': np.array([]), 'labels_train': np.array([]),
                'X_validation': np.array([]), 'labels_validation': np.array([]),
                'X_test': np.array([]), 'labels_test': np.array([])
            }
            continue
        X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        print(f"Expérience {expId} - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        dataSplit[expId] = {
            'X_train': X_train, 'labels_train': y_train,
            'X_validation': X_val, 'labels_validation': y_val,
            'X_test': X_test, 'labels_test': y_test
        }
    return dataSplit

def newSplitData(data, experiments):
    dataSplit = {}
    for expId in experiments:
        expData = data[expId]
        X = expData['X']
        labels = expData['labels']
        print(f"Expérience {expId} - X shape: {X.shape}, labels shape: {labels.shape}")

        if X.shape[0] == 0 or labels.shape[0] == 0:
            print(f"Expérience {expId} - Pas de données")
            dataSplit[expId] = {'X': np.array([]), 'labels': np.array([])}
            continue
        # Pour validation croisée, on garde tout ensemble.
        dataSplit[expId] = {'X': X, 'labels': labels}
        print(f"Expérience {expId} - Données préparées pour validation croisée : {X.shape}, Labels : {labels.shape}")
    
    return dataSplit