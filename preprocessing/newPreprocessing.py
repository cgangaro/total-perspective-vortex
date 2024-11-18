from typing import List
import mne
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from errorClasses import PreprocessingError
import matplotlib
import matplotlib.pyplot as plt

class NewPreprocessing:

    @staticmethod
    def loadRawFile(subject: int, run: int):
        # Charger les données pour un sujet et un run spécifique
        raw_fnames = eegbci.load_data(subject, runs=[run])
        raw_files = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
        raw = mne.concatenate_raws(raw_files)
        # Standardiser les annotations pour que tous les runs aient les mêmes event_id
        eegbci.standardize(raw)
        return raw
    
    @staticmethod
    def load_all_data(subjects, runs):
        raw_files = []
        for subject in subjects:
            for run in runs:
                raw = NewPreprocessing.loadRawFile(subject, run)
                raw_files.append(raw)
        # Concaténer tous les fichiers raw
        raw = mne.concatenate_raws(raw_files)
        return raw
