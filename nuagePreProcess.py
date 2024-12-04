import mne
import logging
import numpy as np
from mne import pick_types
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)

DATASET_PATH = "/home/cgangaro/goinfre/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/"

good_channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]

# Ex:
# mapping = {0: 'Rest', 1: 'Left fist', 2: 'Right fist'}

def preprocessOneExperiment(subjects, runs, mapping, balance=False, average=False):
    epochs, event_id = read_dataset(subjects, runs, mapping)

    if balance:
        epochs = balanceClasses(epochs)

    if average:
        xTrainAvg, yTrainAvg = averageOverEpochs(epochs, event_id)
    else:
        xTrainAvg = epochs
        yTrainAvg = epochs.events[:, -1]
    
    return xTrainAvg, yTrainAvg

def read_dataset(subjects, runs, mapping):
    raws = []

    for sub in subjects:
        for run in runs:
            filePath = DATASET_PATH + f"S{sub:03d}/S{sub:03d}R{run:02d}.edf"
            # print(f"Reading file: {filePath}")
            raw = mne.io.read_raw_edf(filePath, preload=True)
            if raw.info['sfreq'] != 160.0:
                raw.resample(sfreq=160.0)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage("standard_1005")

            events, _ = mne.events_from_annotations(
                raw=raw,
                event_id=dict(T1=1, T2=2)
            )
            # mapping = get_mapping(r[0])
            annotations = mne.annotations_from_events(
                events=events,
                event_desc=mapping,
                sfreq=raw.info["sfreq"]
            )
            raw.set_annotations(annotations)
            raws.append(raw)

    raw = mne.concatenate_raws(raws)

    channels = raw.info["ch_names"]
    bad_channels = [x for x in channels if x not in good_channels]
    raw.drop_channels(bad_channels)

    events, event_id = mne.events_from_annotations(raw)

    tmin = -.500
    tmax = 1.000
    baseline = (None, 0)
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")

    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        picks=picks,
        proj=True,
        preload=True
    )

    return epochs, event_id

def balanceClasses(epochs):
    event_id = epochs.event_id
    keys = list(event_id.keys())
    lenKey0 = len(epochs[keys[0]])
    lenKey1 = len(epochs[keys[1]])
    print(f"Epochs for eventId {event_id[keys[0]]}: {lenKey0}, eventId {event_id[keys[1]]}: {lenKey1}")
    if lenKey0 < lenKey1:
        epochs = setEpochsClassSize(epochs, event_id[keys[1]], lenKey0)
    elif lenKey1 < lenKey0:
        epochs = setEpochsClassSize(epochs, event_id[keys[0]], lenKey1)
    else:
        print("Classes are already balanced")
        return epochs
    print(f"Epochs for eventId {event_id[keys[0]]}: {lenKey0}, eventId {event_id[keys[1]]}: {lenKey1}")
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
    print(f"keys: {keys}")

    epochsSize = min(len(epochs[keys[0]]), len(epochs[keys[1]]))

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