
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from mne.datasets import eegbci
from mne.channels import make_standard_montage


class Preprocessing:
    
    @staticmethod
    def makeMontage(raw, montageType="standard_1005", display=False):
        # MONTAGE
        eegbci.standardize(raw)
        montage = make_standard_montage(montageType)
        raw.set_montage(montage, on_missing='ignore')
        montage = raw.get_montage()
        if display:
            montage.plot()
        return raw

    @staticmethod
    def filterRawData(raw, low_cutoff, high_cutoff, notch_freq):
        # NOTCH FILTER
        # Notch Filter permet de supprimer les fréquences parasites. Ici, on supprime les fréquences de 60 Hz
        # car elles sont souvent dues à des interférences électriques. Au USA, la fréquence électrique est de 60 Hz,
        # alors qu'en Europe, elle est de 50 Hz.
        raw.notch_filter(notch_freq, fir_design='firwin')

        # BANDPASS FILTER
        # Un filtre passe-bande permet de ne conserver que les fréquences qui nous intéressent. Ici, on ne garde que
        # les fréquences entre 8 et 40 Hz.
        rawFiltered = raw.filter(l_freq=low_cutoff, h_freq=high_cutoff, fir_design='firwin', skip_by_annotation='edge')
        return rawFiltered
    
    @staticmethod
    def displayPSD(raw, titlePrefix=""):
        # POWER SPECTRAL DENSITY
        psd = raw.compute_psd()
        # fig1, ax1 = plt.subplots(figsize=(10, 6))
        # fig2, ax2 = plt.subplots(figsize=(10, 6))
        # psd.plot(average=False, ax=ax1)
        # ax1.set_title(titlePrefix + " - " + "Power Spectral Density")
        # psd.plot(average=True, ax=ax2)
        # ax2.set_title(titlePrefix + " - " + "Power Spectral Density - Moyenne")
        fig1 = psd.plot(average=False)
        fig1.suptitle(titlePrefix + " - Power Spectral Density")  # Ajoute un titre au graphique
        fig2 = psd.plot(average=True)
        fig2.suptitle(titlePrefix + " - Power Spectral Density - Moyenne")  # Ajoute un titre au graphique
        plt.show()

    @staticmethod
    def visualizeRawData(raw, rawFiltered, nbChannels=5, filteringRange="8-40Hz"):
        rawBeforeFilter = raw.copy()
        picks = mne.pick_types(raw.info, eeg=True)[:5]
        dataBeforeFilter, times = rawBeforeFilter[picks, :]
        dataAfterFilter, _ = rawBeforeFilter[picks, :]
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        axs[0].plot(times, dataBeforeFilter.T, color='blue')
        axs[0].set_title("EEG Data Before Filtering")
        axs[0].set_ylabel("Amplitude (uV)")
        axs[1].plot(times, dataAfterFilter.T, color='green')
        axs[1].set_title("EEG Data After Filtering " + filteringRange)
        axs[1].set_xlabel("Temps (s)")
        axs[1].set_ylabel("Amplitude (uV)")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def loadPreprocessedData(subjects, experiments, data_dir='data'):
        data = {}
        for exp_id in experiments:
            nbSubjects = len(subjects)
            file_path = os.path.join(data_dir, f"experiment_{exp_id}_n{nbSubjects}.npz")
            if not os.path.exists(file_path):
                print(f"Le fichier {file_path} n'existe pas.")
                data[exp_id] = {'X': np.array([]), 'labels': np.array([])}
                continue
            loaded = np.load(file_path, allow_pickle=True)
            data_exp = loaded['arr_0'].item()
            data[exp_id] = data_exp
            print(f"Données chargées pour l'expérience {exp_id}: {data_exp['X'].shape[0]} échantillons")
        return data
