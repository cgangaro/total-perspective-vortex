import numpy as np
import mne
import os
from preprocessing.preprocessing import Preprocessing
from preprocessing.newPreprocessing import NewPreprocessing
from preprocessing.featureExtraction import FeatureExtraction
from dataclasses import dataclass


@dataclass
class Configuration:
    ICA: bool
    EOG: bool
    montageKind: str
    lowCutoff: float
    highCutoff: float
    notchFreq: int
    nIcaComponents: int
    CSPTransformerNComponents: int


def getData(subjects, experiments, config: Configuration, save_to_file=False, output_dir='data', name='default'):
    data_per_experiment = {}
    for exp_id, runs in experiments.items():
        X_list = []
        labels_list = []
        groups = []
        print(f"Traitement de l'expérience {exp_id} avec les runs {runs}")
        if len(runs) == 0:
            print(f"  Pas de runs pour l'expérience {exp_id}")
            continue
        for subject in subjects:
            print(f"  Sujet {subject}")
            try:
                # Charger les données pour le sujet et les runs de 
                # cette expérience
                raw, events, event_id = NewPreprocessing.load_all_data([subject], runs)
            except Exception as e:
                print(f"    Impossible de charger les données pour le sujet {subject} et les runs {runs}: {e}")
                continue
            # Appliquer le montage
            raw = Preprocessing.makeMontage(
                raw,
                montageType=config.montageKind,
                display=False
            )
            # Filtrer les données
            rawFiltered = Preprocessing.filterRawData(
                raw,
                low_cutoff=config.lowCutoff,
                high_cutoff=config.highCutoff,
                notch_freq=config.notchFreq
            )
            # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_id)
            if config.ICA:
                rawFiltered = FeatureExtraction.ICAProcess(
                    rawFiltered,
                    config.nIcaComponents,
                    config.EOG
                )
            # Créer les epochs
            events, event_id = mne.events_from_annotations(rawFiltered)
            tmin, tmax = -1., 4.
            epochs = mne.Epochs(rawFiltered, events, event_id=event_id, tmin=tmin, tmax=tmax,
                                baseline=None, preload=True)
            # Inclure les classes spécifiques si nécessaire
            epochs = epochs[['T0', 'T1', 'T2']]
            # Récupérer les labels
            labels = epochs.events[:, -1] - 1  # Labels : [0, 1, 2]
            X = epochs.get_data().astype(np.float64)
            if len(labels) == 0:
                print(f"    Pas d'epochs pour le sujet {subject} dans l'expérience {exp_id}")
                continue  # Passer au sujet suivant s'il n'y a pas d'epochs
            # Ajouter les données et les labels aux listes
            X_list.append(X)
            labels_list.append(labels)
            groups.extend([subject] * len(labels))
        if len(X_list) == 0:
            print(f"Aucune donnée prétraitée pour l'expérience {exp_id}.")
            continue
        # Concaténer les données
        X_exp = np.concatenate(X_list)
        labels_exp = np.concatenate(labels_list)
        groups_exp = np.array(groups)
        # Sauvegarder les données pour cette expérience
        data_per_experiment[exp_id] = {'X': X_exp, 'labels': labels_exp, 'groups': groups_exp}
        print(f"Données prétraitées pour l'expérience {exp_id} : {X_exp.shape[0]} échantillons")

    if save_to_file:
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"experiment_{name}.npz")
        np.savez(output_path, data_per_experiment)
        print(f"Données enregistrées pour l'expérience {exp_id} dans {output_path}")

    return data_per_experiment



