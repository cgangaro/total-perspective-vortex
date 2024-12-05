import mne
import logging
import numpy as np
import matplotlib.pyplot as plt
from mne import Annotations, make_fixed_length_events, pick_types
from sklearn.model_selection import train_test_split

from utils.dataclassModels import PreProcessConfiguration
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)

good_channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]


def preprocessOneExperiment(subjects, runs, mapping, config: PreProcessConfiguration, balance=False, average=False, splitData=False):
    if runs == [1, 2]:
        # openEpochs, open_event_id = read_dataset(subjects, [1], {0: 'Eyes open'}, config, eyesOpenClosed=True)
        # epochs, event_id = read_dataset([1, 2], [1, 2], {0: 'Eyes closed'}, config, eyesOpenClosed=True)
        epochs, event_id = preprocessOneSubjectEyesOpenClosed(subjects, runs, config)
        print(f"Epochs: {len(epochs)}")
        # print(f"Event_id: {event_id}")
        # print(f"Event_id keys: {event_id.keys()}")
        # print(f"Event_id values: {event_id.values()}")
        # exit()
    else:
        epochs, event_id = read_dataset(subjects, runs, mapping, config)

    if splitData:
        print(f"Splitting data into train and test sets - {len(epochs)} epochs")
        indices = np.arange(len(epochs))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

        epochsTrain = epochs[train_indices]
        epochsTest = epochs[test_indices]
        print(f"Train set: {len(epochsTrain)} epochs, Test set: {len(epochsTest)} epochs")
    else:
        epochsTrain = epochs

    if balance:
        epochsTrain = balanceClasses(epochsTrain)

    if average:
        xTrainAvg, yTrainAvg = averageOverEpochs(epochsTrain, event_id)
    else:
        xTrainAvg = epochsTrain
        yTrainAvg = epochsTrain.events[:, -1]

    if splitData:
        xTest = epochsTest
        yTest = epochsTest.events[:, -1]
        return xTrainAvg, yTrainAvg, xTest, yTest
    else:
        return xTrainAvg, yTrainAvg

def read_dataset(subjects, runs, mapping, config: PreProcessConfiguration, display=False, eyesOpenClosed=False):
    raws = []
    print(f"Reading dataset for {len(subjects)} subjects and runs {runs}")

    for i, sub in enumerate(subjects):
        for run in runs:

            filePath = config.dataLocation + f"S{sub:03d}/S{sub:03d}R{run:02d}.edf"
            raw = mne.io.read_raw_edf(filePath, preload=True)

            if raw.info['sfreq'] != 160.0:
                raw.resample(sfreq=160.0)

            mne.datasets.eegbci.standardize(raw)

            if config.makeMontage:
                raw.set_montage(config.montageShape)
                if display:
                    biosemi_montage = mne.channels.make_standard_montage(config.montageShape)
                    biosemi_montage.plot()
                    plt.show()

            raw.notch_filter(60, method="iir")

            raw.filter(config.lowFilter, config.highFilter, fir_design="firwin",
                        skip_by_annotation="edge")

            events, _ = mne.events_from_annotations(
                    raw=raw,
                    event_id=dict(T1=1, T2=2)
                )
            annotations = mne.annotations_from_events(
                events=events,
                event_desc=mapping,
                sfreq=raw.info["sfreq"]
            )

            raw.set_annotations(annotations)

            raws.append(raw)

        if (i+1) % 10 == 0:
            print(f"{i+1}/{len(subjects)} subjects read")

    raw = mne.concatenate_raws(raws)

    channels = raw.info["ch_names"]
    bad_channels = [x for x in channels if x not in good_channels]
    raw.drop_channels(bad_channels)

    events, event_id = mne.events_from_annotations(raw)

    baseline = (None, 0)
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")

    tmin = config.epochsTmin
    tmax = config.epochsTmax
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        proj=True,
        preload=True
    )
    # mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_id)
    print(f"Dataset read: {len(epochs)} epochs")
    return epochs, event_id

def balanceClasses(epochs):
    event_id = epochs.event_id
    keys = list(event_id.keys())
    lenKey0 = len(epochs[keys[0]])
    lenKey1 = len(epochs[keys[1]])
    print(f"Balance Classes - Epochs for eventId {event_id[keys[0]]}: {lenKey0}, eventId {event_id[keys[1]]}: {lenKey1}")
    if lenKey0 < lenKey1:
        epochs = setEpochsClassSize(epochs, event_id[keys[1]], lenKey0)
    elif lenKey1 < lenKey0:
        epochs = setEpochsClassSize(epochs, event_id[keys[0]], lenKey1)
    else:
        print("Classes are already balanced")
        return epochs
    return epochs

def setEpochsClassSize(epochs, eventId, size):
    if len(epochs) < size:
        print("Not enough epochs to balance classes")
        return epochs
    event_indices = np.where(epochs.events[:, -1] == eventId)[0]
    if len(event_indices) <= size:
        print(f"Epochs for eventId {eventId} already have size {len(event_indices)} ≤ target size {size}.")
        return epochs
    excess_indices = np.random.choice(event_indices, size=len(event_indices) - size, replace=False)
    print(f"Reducing size of eventId {eventId} from {len(event_indices)} to {size} by dropping {len(excess_indices)} epochs.")

    epochs.drop(excess_indices)
    return epochs

def averageOverEpochs(epochs, event_id, epochsSizeForAverage = 30):
    newEpochs = []
    newLabels = []

    keys = list(event_id.keys())

    epochsSize = min(len(epochs[keys[0]]), len(epochs[keys[1]]))

    print(f"Average over epochs - Epochs for eventId {event_id[keys[0]]}: {len(epochs[keys[0]])}, eventId {event_id[keys[1]]}: {len(epochs[keys[1]])}")

    i = 0
    while i < epochsSize:
        if i + epochsSizeForAverage > epochsSize:
            epochsSizeForAverage = epochsSize - i
        class0Avg = epochs[keys[0]][i:i+epochsSizeForAverage].average().get_data()
        class1Avg = epochs[keys[1]][i:i+epochsSizeForAverage].average().get_data()
        newEpochs.append(class0Avg)
        newLabels.append(event_id[keys[0]])
        newEpochs.append(class1Avg)
        newLabels.append(event_id[keys[1]])
        i += epochsSizeForAverage

    return np.array(newEpochs), np.array(newLabels)


def preprocessOneSubjectEyesOpenClosed(subjects, runs, config: PreProcessConfiguration, display=False, eyesOpenClosed=False):
    raws = []
    print(f"preprocessOneSubjectEyesOpenClosed Reading dataset for {len(subjects)} subjects and runs {runs}")

    all_epochs = []
    all_labels = []
    for i, sub in enumerate(subjects):
        for run in runs:

            filePath = config.dataLocation + f"S{sub:03d}/S{sub:03d}R{run:02d}.edf"
            raw = mne.io.read_raw_edf(filePath, preload=True)

            raw.rename_channels(lambda x: x.strip('.'))
            if raw.info['sfreq'] != 160.0:
                raw.resample(sfreq=160.0)

            mne.datasets.eegbci.standardize(raw)

            if config.makeMontage:
                raw.set_montage(config.montageShape)
                if display:
                    biosemi_montage = mne.channels.make_standard_montage(config.montageShape)
                    biosemi_montage.plot()
                    plt.show()
            # Filtrage
            # raw.filter(config.lowFilter, config.highFilter, fir_design='firwin', skip_by_annotation='edge')

            # Sélection des canaux EEG
            picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')


            # Détermination de la description en fonction du run
            if run == 1:
                description = 'Eyes Open'
                event_id = {'Eyes Open': 0}
            elif run == 2:
                description = 'Eyes Closed'
                event_id = {'Eyes Closed': 1}
            else:
                description = 'Unknown'

            # Création des événements à intervalles réguliers
            epoch_duration = 1.0  # Durée de chaque epoch en secondes
            overlap = 0.0         # Chevauchement entre les epochs en secondes
            events = make_fixed_length_events(raw, start=0, stop=None, duration=epoch_duration - overlap, overlap=overlap, first_samp=True, id=0)

            # Mise à jour des identifiants d'événements en fonction de la condition
            events[:, 2] = 0 if description == 'Eyes Open' else 1

                # Création des epochs
            tmin = 0.0
            tmax = epoch_duration
            # good_channels = ["Cz", "C1", "C2", "C3", "C4", "FC3", "FC4", "CP3", "CP4"]O1, Oz, O2, PO7, PO3, POz, PO4, et PO8 Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8
            # good_channels = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]
            # channels = raw.info["ch_names"]
            # print(f"Channels: {channels}")
            # bad_channels = [x for x in channels if x not in good_channels]
            # print(f"Bad channels: {bad_channels}")
            # raw.drop_channels(bad_channels)
            # print(f"Channels after dropping: {raw.info['ch_names']}")
            # picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

            # Filtrer les epochs pour la condition actuelle
            epochs = epochs[description]

            labels = epochs.events[:, -1]
            # print(f"Subject {subject}, Run {run}: {len(epochs)} epochs, unique labels: {np.unique(labels)}")

            all_epochs.append(epochs)
            all_labels.extend(labels)

    # Concaténer les epochs et les labels des deux runs
    epochs = mne.concatenate_epochs(all_epochs)
    labels = np.array(all_labels)
    event_id = {'Eyes Open': 0, 'Eyes Closed': 1}

    return epochs, event_id