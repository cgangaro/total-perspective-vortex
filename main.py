import mne
from mne.datasets import eegbci
import matplotlib
import matplotlib.pyplot as plt
from preprocessing.preprocessing import Preprocessing
from preprocessing.newPreprocessing import NewPreprocessing
from preprocessing.featureExtraction import FeatureExtraction
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from cspTransformer import CSPTransformer
from dataclasses import dataclass
import numpy as np
matplotlib.use("webagg")


@dataclass
class Configuration:
    ICA: bool
    EOG: bool


def main():
    try:
        config = Configuration(ICA=False, EOG=False)
        subjects = [1, 2, 3, 4, 5, 6]
        runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        montage_kind = "standard_1005"
        # montage_kind = "biosemi64"
        low_cutoff = 8.
        high_cutoff = 40.
        notch_freq = 60
        n_ica_components = 20
        # experiments = {
        #     0: [1, 2],     # Baseline
        #     1: [3, 7, 11], # Task 1
        #     2: [4, 8, 12], # Task 2
        #     3: [5, 9, 13], # Task 3
        #     4: [6, 10, 14],# Task 4
        # }
        # all_scores = []

        X_list = []
        labels_list = []
        groups = []

        for subject in subjects:
            # Charger les données pour le sujet
            raw, events, event_id = NewPreprocessing.load_all_data([subject], runs)
            print(f"Sujet {subject} - Event ID: {event_id}")
            print(f"Sujet {subject} - Nombre d'événements: {len(events)}")

            # Appliquer le montage
            raw = Preprocessing.makeMontage(raw, montage_kind, False)
            # original_raw.plot(n_channels=5, scalings='auto', title='Données EEG brutes', show=True, block=True)
            # Preprocessing.displayPSD(original_raw, "Donnees EEG brutes")

            # Filtrer les données
            rawFiltered = Preprocessing.filterRawData(raw, low_cutoff, high_cutoff, notch_freq)
            # rawFiltered.plot(n_channels=5, scalings='auto', title='Données EEG filtrées', show=True)
            # Preprocessing.displayPSD(rawFiltered, "Donnees EEG filtrees")
            # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_id)
            if config.ICA:
                rawFiltered = FeatureExtraction.ICAProcess(rawFiltered, n_ica_components, config.EOG)

            # Créer les epochs
            tmin, tmax = -1., 4.
            epochs = mne.Epochs(rawFiltered, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)
            # # Inclure toutes les classes : 'T0', 'T1', 'T2'
            # event_ids_to_keep = ['T0', 'T1', 'T2']
            # epochs = epochs[event_ids_to_keep]

            # Récupérer les labels
            labels = epochs.events[:, -1] - 1  # Labels : [0, 1, 2]
            X = epochs.get_data()  # Forme : (n_epochs, n_channels, n_times)
            X = X.astype(np.float64)

            # Ajouter les données et les labels aux listes
            X_list.append(X)
            labels_list.append(labels)
            groups.extend([subject] * len(labels))  # Ajouter le numéro du sujet pour chaque epoch

        # Concaténer les données de tous les sujets
        X = np.concatenate(X_list)
        labels = np.concatenate(labels_list)
        groups = np.array(groups)

        print("Shape des données :", X.shape)
        print("Shape des labels :", labels.shape)
        print("Shape des groupes :", groups.shape)
        print("Labels uniques :", np.unique(labels))
        print("Groupes uniques :", np.unique(groups))

        # Définition du classifieur
        csp_pipeline = Pipeline([
            ('csp', CSPTransformer(n_components=4)),
            ('lda', LDA())
        ])

        # Validation croisée LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        scores = cross_val_score(csp_pipeline, X, labels, cv=logo, groups=groups, scoring='accuracy', n_jobs=-1, error_score='raise')
        print(f"Précision moyenne sur tous les sujets : {scores.mean():.2f} ± {scores.std():.2f}")

        return
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
