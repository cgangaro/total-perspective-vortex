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
from sklearn.model_selection import train_test_split
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)

# def preprocessOneSubjectRealOrImaginary(subject, config: PreProcessConfiguration):
#     real_runs = [3, 7, 11]
#     imaginary_runs = [5, 9, 13]
#     all_runs = real_runs + imaginary_runs
#     print(f"Preprocessing subject {subject} for runs {all_runs} - Real or Imaginary")
#     raw_fnames = eegbci.load_data(subject, runs=all_runs, verbose="ERROR")
#     all_epochs = []
#     all_labels = []
#     for fname, run in zip(raw_fnames, all_runs):
#         raw = mne.io.read_raw_edf(fname, preload=True, stim_channel='auto')
#         raw.rename_channels(lambda x: x.strip('.'))
#         eegbci.standardize(raw)
#         if config.makeMontage:
#             raw.set_montage(config.montageShape)
        
#         # Filtrage
#         raw.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')

#         # Sélection des canaux EEG
#         picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

#         # ICA pour éliminer les artefacts
#         if config.ica:
#             ica = mne.preprocessing.ICA(n_components=config.icaComponents, random_state=97, max_iter=800)
#             ica.fit(raw, picks=picks)

#             if config.eog:
#                 eog_channels = ['Fp1', 'Fp2', 'Fpz', 'AFz', 'AF3', 'AF4']
#                 eog_chs_present = [ch for ch in eog_channels if ch in raw.ch_names]
#                 if eog_chs_present:
#                     eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_chs_present)
#                     ica.exclude = eog_indices
#                     raw = ica.apply(raw)
#                 else:
#                     print("No frontal channels found for EOG artifact detection. Skipping EOG artifact removal.")

#         # Rééchantillonnage
#         if config.resample:
#             raw.resample(config.resampleFreq, npad="auto")

#         # Détermination de la description en fonction du run
#         if run in real_runs:
#             description = 'Real'
#             event_id = {'Real': 0}
#             eventsId = dict(T1=0,T2=0)
#         elif run in imaginary_runs:
#             description = 'Imaginary'
#             event_id = {'Imaginary': 1}
#             eventsId = dict(T1=1,T2=1)
#         else:
#             description = 'Unknown'

#         events, _ = events_from_annotations(raw, event_id=eventsId)
#         # # Création des événements à intervalles réguliers
#         # epoch_duration = 1.0  # Durée de chaque epoch en secondes
#         # overlap = 0.0         # Chevauchement entre les epochs en secondes
#         # events = make_fixed_length_events(raw, start=0, stop=None, duration=epoch_duration - overlap, overlap=overlap, first_samp=True, id=0)

#         # # Mise à jour des identifiants d'événements en fonction de la condition
#         # events[:, 2] = 0 if description == 'Real' else 1

        
#         # Création des epochs
#         epochs = Epochs(raw, events, event_id, config.epochsTmin, config.epochsTmax, proj=True, picks=picks, baseline=None, preload=True)

#         # Filtrer les epochs pour la condition actuelle
#         epochs = epochs[description]

#         labels = epochs.events[:, -1]
#         print(f"Subject {subject}, Run {run}: {len(epochs)} epochs, unique labels: {np.unique(labels)}")

#         all_epochs.append(epochs)
#         all_labels.extend(labels)

#     # Concaténer les epochs et les labels des deux runs
#     epochs = mne.concatenate_epochs(all_epochs)
#     labels = np.array(all_labels)

#     return epochs, labels


# def preprocessRestVsMovement(subject, config: PreProcessConfiguration):
#     """
#     Prétraitement pour classifier repos (T0) vs mouvement (T1/T2) en utilisant les bandes alpha et beta.
#     """
#     runs = [3, 4, 5, 6, 7, 8, 9, 10]  # Runs liés aux mouvements
#     print(f"Preprocessing subject {subject} for runs {runs} - Rest vs Movement")
    
#     # Charger les données brutes
#     raw_fnames = eegbci.load_data(subject, runs=runs, verbose="ERROR")
#     raw_list = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
#     raw = mne.concatenate_raws(raw_list)
#     raw.rename_channels(lambda x: x.strip('.'))
#     eegbci.standardize(raw)
    
#     # Montage
#     if config.makeMontage:
#         raw.set_montage(config.montageShape)
    
#     # Filtrage passe-bande pour les bandes alpha et beta
#     raw_alpha = raw.copy().filter(8, 13, fir_design='firwin', skip_by_annotation='edge')  # Bande alpha
#     raw_beta = raw.copy().filter(13, 30, fir_design='firwin', skip_by_annotation='edge')  # Bande beta
    
#     # Sélection des canaux EEG
#     picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
#     # ICA pour éliminer les artéfacts
#     if config.ica:
#         ica = mne.preprocessing.ICA(n_components=config.icaComponents, random_state=97, max_iter=800)
#         ica.fit(raw, picks=picks)
        
#         if config.eog:
#             eog_channels = ['Fp1', 'Fp2', 'Fpz', 'AFz', 'AF3', 'AF4']
#             eog_chs_present = [ch for ch in eog_channels if ch in raw.ch_names]
#             if eog_chs_present:
#                 eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_chs_present)
#                 ica.exclude = eog_indices
#                 raw = ica.apply(raw)
#             else:
#                 print("No frontal channels found for EOG artifact detection. Skipping EOG artifact removal.")
    
#     # Rééchantillonnage
#     if config.resample:
#         raw.resample(config.resampleFreq, npad="auto")
    
#     # Extraction des événements
#     event_id = {'T0': 0, 'T1': 1, 'T2': 1}  # T1 et T2 regroupés comme mouvement
#     events, _ = events_from_annotations(raw, event_id=event_id)
    
#     # Création des epochs pour les bandes alpha et beta
#     # epochs_alpha = Epochs(raw_alpha, events, event_id=event_id, tmin=-0.5, tmax=2.,
#     #                       proj=True, picks=picks, baseline=None, preload=True)
#     epochs_beta = Epochs(raw_beta, events, event_id=event_id, tmin=-0.5, tmax=2.,
#                          proj=True, picks=picks, baseline=None, preload=True)
    
#     # Combiner les bandes alpha et beta en tant que caractéristiques
#     # epochs_data_alpha = epochs_alpha.get_data()
#     # epochs_data_beta = epochs_beta.get_data()
#     # combined_epochs_data = np.concatenate([epochs_data_alpha, epochs_data_beta], axis=1)
    
#     # Récupération des étiquettes
#     labels = epochs_beta.events[:, -1]
    
#     print(f"Subject {subject}: {len(epochs_beta)} epochs, labels: {len(labels)}, unique labels: {np.unique(labels)}")
    
#     return epochs_beta, labels

# def preprocessOneSubjectOneExperimentAAA(subject, config: PreProcessConfiguration):
#     runs = [3, 4, 5, 6, 7, 8, 9, 10]
#     print(f"Preprocessing AAA subject {subject} for runs {runs}")
#     rawFnames = eegbci.load_data(subject, runs=runs, verbose="ERROR", path=config.dataLocation)
#     rawBrut = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in rawFnames]
#     rawBrutConcat = mne.concatenate_raws(rawBrut, verbose="ERROR")

#     rawBrutConcat.rename_channels(lambda x: x.strip('.'))
#     eegbci.standardize(rawBrutConcat)

#     if config.makeMontage:
#         rawBrutConcat.set_montage(config.montageShape)
    
#     # rawBrutConcat.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')
#     # raw_alpha = raw.copy().filter(8, 13, fir_design='firwin', skip_by_annotation='edge')  # Bande alpha
#     rawBrutConcat.filter(8, 13, fir_design='firwin', skip_by_annotation='edge')

#     picks = pick_types(rawBrutConcat.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

#     if config.resample:
#         rawBrutConcat.resample(config.resampleFreq, npad="auto")

#     eventsId = dict(T1=0,T2=1)
#     events, _ = events_from_annotations(rawBrutConcat, event_id=eventsId)

#     epochs = Epochs(rawBrutConcat, events, eventsId, config.epochsTmin, config.epochsTmax, proj=True, picks=picks, baseline=None, preload=True)
#     labels = epochs.events[:, -1]
#     return epochs, labels

def preprocessOneSubjectOneExperimentSpecial(subject, config: PreProcessConfiguration, ica=None, display=False):
    
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