import numpy as np
import mne
import os
from preprocessing.preprocessing import Preprocessing
from preprocessing.newPreprocessing import NewPreprocessing
from preprocessing.featureExtraction import FeatureExtraction
from dataclasses import dataclass


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
                raw, events, event_id = NewPreprocessing.load_all_data([subject], runs)
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

            # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_id)

            if config.preProcess.ICA:
                rawFiltered = FeatureExtraction.ICAProcess(
                    rawFiltered,
                    config.preProcess.nIcaComponents,
                    config.preProcess.EOG
                )

            # EPOCHS
            events, event_id = mne.events_from_annotations(rawFiltered)
            tmin, tmax = -1., 4.
            epochs = mne.Epochs(
                rawFiltered,
                events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                preload=True
            )

            if config.preProcess.useRestMoment:
                epochs = epochs[['T0', 'T1', 'T2']]
                labels = epochs.events[:, -1] - 1
            else:
                epochs = epochs[['T1', 'T2']]
                labels = epochs.events[:, -1] - 2

            X = epochs.get_data().astype(np.float64)
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
