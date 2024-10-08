from typing import List
import mne
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from errorClasses import PreprocessingError
from mne.preprocessing import ICA

class Preprocessing:
    @staticmethod
    def initRawFiles(subjects: List[int]):
        # real_execution = [3, 5, 7, 9, 11, 13]
        # imaginary_execution = [4, 6, 8, 10, 12, 14]
        real_execution = [5]
        imaginary_execution = [6]
        print("Initializing raw files with subjects: ", subjects)
        raw_files = []
        for subject in subjects:
            for run in real_execution:
                event_mapping = {1: 'rest', 2: 'do/feet', 3: 'do/hands'}
                raw_file = Preprocessing.loadRawFile(subject, run, event_mapping)
                print("Raw file loaded for subject ", subject, " and run ", run)
                raw_files.append(raw_file)
            for run in imaginary_execution:
                event_mapping = {1: 'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
                raw_file = Preprocessing.loadRawFile(subject, run, event_mapping)
                print("Raw file loaded for subject ", subject, " and run ", run)
                raw_files.append(raw_file)
        print("ok")
        print(raw_files)
        raw = mne.io.concatenate_raws(raw_files)
        print(raw)
        events, event_dict = mne.events_from_annotations(raw)
        print(event_dict)
        return raw
    
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
