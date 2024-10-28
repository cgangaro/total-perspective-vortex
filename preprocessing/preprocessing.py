from typing import List
import mne
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from errorClasses import PreprocessingError
import matplotlib
import matplotlib.pyplot as plt

class Preprocessing:

    @staticmethod
    def initRawFiles2(subjects: List[int], executions: List[int]):
        # Tâches séparées pour les mouvements réels et imaginés
        run_execution = [3, 5, 7, 9, 11, 13]
        run_imagery = [4, 6, 8, 10, 12, 14]

        raw_files = []

        for subject in subjects:
            for exec_run, imag_run in zip(run_execution, run_imagery):
                # Charger les données EEG pour les tâches réelles
                raw_files_execution = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in
                                       eegbci.load_data(subject, exec_run)]
                raw_execution = mne.concatenate_raws(raw_files_execution)

                # Charger les données EEG pour les tâches imaginées
                raw_files_imagery = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in
                                     eegbci.load_data(subject, imag_run)]
                raw_imagery = mne.concatenate_raws(raw_files_imagery)

                # Extraire les événements et créer des annotations pour les tâches réelles
                events, _ = mne.events_from_annotations(raw_execution, event_id=dict(T0=1, T1=2, T2=3))
                mapping_execution = {1: 'rest', 2: 'do/feet', 3: 'do/hands'}
                annot_execution = mne.annotations_from_events(
                    events=events, event_desc=mapping_execution, sfreq=raw_execution.info['sfreq'],
                    orig_time=raw_execution.info['meas_date'])
                raw_execution.set_annotations(annot_execution)

                # Extraire les événements et créer des annotations pour les tâches imaginées
                events, _ = mne.events_from_annotations(raw_imagery, event_id=dict(T0=1, T1=2, T2=3))
                mapping_imagery = {1: 'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
                annot_imagery = mne.annotations_from_events(
                    events=events, event_desc=mapping_imagery, sfreq=raw_imagery.info['sfreq'],
                    orig_time=raw_imagery.info['meas_date'])
                raw_imagery.set_annotations(annot_imagery)

                # Ajouter les fichiers traités à la liste
                raw_files.append(raw_execution)
                raw_files.append(raw_imagery)

        # Concaténer les fichiers bruts pour tous les sujets et exécutions
        raw = mne.io.concatenate_raws(raw_files)

        # Extraire les événements et les dictionnaires d'événements
        events, event_dict = mne.events_from_annotations(raw)

        return raw, events, event_dict
    
    @staticmethod
    def initRawFiles(subjects: List[int], executions: List[int]):
        real_execution = [3, 5, 7, 9, 11, 13]
        imaginary_execution = [4, 6, 8, 10, 12, 14]
        one_fist_execution = [3, 4, 7, 8, 11, 12]
        both_fists_or_feet_execution = [5, 6, 9, 10, 13, 14]

        all_executions = sorted(executions)
        print("All executions: ", all_executions)
        # real_execution = [5]
        # imaginary_execution = [6]
        print("Initializing raw files with subjects: ", subjects)
        real_one_event_mapping = {1: 'rest', 2: 'do/left_fist', 3: 'do/right_fist'}
        imaginary_one_event_mapping = {1: 'rest', 2: 'imagine/left_fist', 3: 'imagine/right_fist'}
        real_both_event_mapping = {1: 'rest', 2: 'do/hands', 3: 'do/feet'}
        imaginary_both_event_mapping = {1: 'rest', 2: 'imagine/hands', 3: 'imagine/feet'}
        raw_files = []
        for subject in subjects:
            for run in all_executions:
                if (run in real_execution):
                    if (run in one_fist_execution):
                        event_mapping = real_one_event_mapping
                    else:
                        event_mapping = real_both_event_mapping
                else:
                    if (run in imaginary_execution):
                        if (run in one_fist_execution):
                            event_mapping = imaginary_one_event_mapping
                        else:
                            event_mapping = imaginary_both_event_mapping
                raw_file = Preprocessing.loadRawFile(subject, run, event_mapping)
                print("Raw file loaded for subject ", subject, " and run ", run)
                raw_files.append(raw_file)
            # for run in real_execution:
            #     event_mapping = {1: 'rest', 2: 'do/feet', 3: 'do/hands'}
            #     raw_file = Preprocessing.loadRawFile(subject, run, event_mapping)
            #     print("Raw file loaded for subject ", subject, " and run ", run)
            #     raw_files.append(raw_file)
            # for run in imaginary_execution:
            #     event_mapping = {1: 'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
            #     raw_file = Preprocessing.loadRawFile(subject, run, event_mapping)
            #     print("Raw file loaded for subject ", subject, " and run ", run)
            #     raw_files.append(raw_file)
        print("ok")
        print(raw_files)
        raw = mne.io.concatenate_raws(raw_files)
        print(raw)
        events, event_dict = mne.events_from_annotations(raw)
        print(event_dict)
        return raw, events, event_dict
    
    @staticmethod
    def loadRawFile(subject: int, run: int, event_mapping: dict):
        if (len(event_mapping) != 3):
            raise PreprocessingError("The event mapping must contain 3 keys")
        
        raw_data = eegbci.load_data(subject, run)
        # Load the raw data from the subject and run specified
        raw_file = mne.io.concatenate_raws([mne.io.read_raw_edf(it, preload=True, stim_channel='auto') for it in raw_data])
        # Raw data are stored in different files, so we concatenate them
        # preload = True to load the data into memory, to accelerate the process
        # stim_channel = 'auto' to automatically detect the stim channel, which is the channel that contains the event markers
        print(raw_file)
        events, _ = mne.events_from_annotations(raw_file, event_id=dict(T0=1,T1=2,T2=3))
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=event_mapping, sfreq=raw_file.info['sfreq'],
            orig_time=raw_file.info['meas_date'])
        # sfreq = sampling frequency, it's used here to relate precisely the annotations to the data
        # orig_time = original time of the recording
        raw_file.set_annotations(annot_from_events)
        return raw_file
    
    @staticmethod
    def makeMontage(raw, montageType = "standard_1005", display=False):
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
    
    # @staticmethod
    # def deleteBadEog(raw, picks, method, plotIt=None):
    #     raw_corrected = raw.copy()
    #     n_components = 20

    #     ica = ICA(n_components=n_components, method=method, fit_params=None, random_state=97)

    #     ica.fit(raw_corrected, picks=picks)

    #     [eog_indicies, scores] = ica.find_bads_eog(raw, ch_name='Fpz', threshold=1.5)
    #     ica.exclude.extend(eog_indicies)
    #     ica.apply(raw_corrected, n_pca_components=n_components, exclude=ica.exclude)

    #     if plotIt:
    #         ica.plot_components()
    #         ica.plot_scores(scores, exclude=eog_indicies)

    #         plt.show()

    #     return raw_corrected
