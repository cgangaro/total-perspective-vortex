from typing import List
import mne
from mne.datasets import eegbci
from errorClasses import PreprocessingError

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
                raw_file = eegbci.load_data(subject, run, event_mapping)
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
        # raw_file = mne.io.read_raw_edf(raw_data, preload=True, stim_channel='auto')
        raw_file = mne.io.concatenate_raws([mne.io.read_raw_edf(it, preload=True) for it in raw_data])
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
    def testTest():
        subject = [1] # 1, 4
        # run_execution = [5, 9, 13]
        # run_imagery = [6, 10, 14]
        run_execution = [5]
        run_imagery = [6]

        raw_files = []

        for person_number in subject:
            for i, j in zip(run_execution, run_imagery):
                raw_files_execution = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in eegbci.load_data(person_number, i)]
                raw_files_imagery = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in eegbci.load_data(person_number, j)]
                
                raw_execution = mne.io.concatenate_raws(raw_files_execution)
                print(raw_execution)
                raw_imagery = mne.io.concatenate_raws(raw_files_imagery)
                print(raw_imagery)

                events, _ = mne.events_from_annotations(raw_execution, event_id=dict(T0=1,T1=2,T2=3))
                mapping = {1:'rest', 2: 'do/feet', 3: 'do/hands'}
                annot_from_events = mne.annotations_from_events(
                    events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
                    orig_time=raw_execution.info['meas_date'])
                raw_execution.set_annotations(annot_from_events)
                
                events, _ = mne.events_from_annotations(raw_imagery, event_id=dict(T0=1,T1=2,T2=3))
                mapping = {1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
                annot_from_events = mne.annotations_from_events(
                    events=events, event_desc=mapping, sfreq=raw_imagery.info['sfreq'],
                    orig_time=raw_imagery.info['meas_date'])
                raw_imagery.set_annotations(annot_from_events)
                
                
                raw_files.append(raw_execution)
                raw_files.append(raw_imagery)