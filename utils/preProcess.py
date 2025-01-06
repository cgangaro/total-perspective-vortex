import mne
import logging
import numpy as np
import matplotlib.pyplot as plt
from mne import pick_types
from sklearn.model_selection import train_test_split

from utils.dataclassModels import PreProcessConfiguration
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)

good_channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]


def preprocessOneExperiment(
        subjects,
        runs,
        mapping,
        config: PreProcessConfiguration,
        balance=False,
        average=False,
        splitData=False
):
    epochs, event_id = readDataset(subjects, runs, mapping, config)

    if splitData:
        print(f"Splitting data into train and test sets - "
              f"{len(epochs)} epochs")
        indices = np.arange(len(epochs))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=0.2,
            random_state=42
        )

        epochsTrain = epochs[train_indices]
        epochsTest = epochs[test_indices]
        print(f"Train set: {len(epochsTrain)} epochs, "
              f"Test set: {len(epochsTest)} epochs")
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


def readDataset(
        subjects,
        runs,
        mapping,
        config: PreProcessConfiguration,
        display=False
):
    raws = []
    print(f"Reading dataset for {len(subjects)} subjects and runs {runs}")

    for i, sub in enumerate(subjects):
        for run in runs:

            filePath = (config.dataLocation +
                        f"S{sub:03d}/S{sub:03d}R{run:02d}.edf")
            raw = mne.io.read_raw_edf(filePath, preload=True)

            if display:
                raw.plot()
                plt.gcf().suptitle(f"Raw Signal - Subject {sub}, Run {run}")
                raw.plot_psd(average=False)
                plt.gcf().suptitle(f"PSD (Per Channel) - Subject {sub},"
                                   f" Run {run}")
                raw.plot_psd(average=True)
                plt.gcf().suptitle(f"PSD (Averaged) - Subject {sub},"
                                   f" Run {run}")
                plt.show()

            if raw.info['sfreq'] != 160.0:
                raw.resample(sfreq=160.0)

            mne.datasets.eegbci.standardize(raw)

            if config.makeMontage:
                raw.set_montage(config.montageShape)
                if display:
                    biosemi_montage = mne.channels.make_standard_montage(
                        "standard_1020"
                    )
                    biosemi_montage.plot()
                    plt.gcf().suptitle(f"Montage (Standard 10-20) - Subject"
                                       f" {sub}")
                    plt.show()

            raw.notch_filter(60, method="iir")

            raw.filter(
                config.lowFilter,
                config.highFilter,
                fir_design="firwin",
                skip_by_annotation="edge"
            )

            if display:
                raw.plot()
                plt.gcf().suptitle(f"Raw Signal Filtered - Subject {sub},"
                                   f" Run {run}")
                raw.plot_psd(average=False)
                plt.gcf().suptitle(f"PSD (Per Channel) Filtered - "
                                   f"Subject {sub}, Run {run}")
                raw.plot_psd(average=True)
                plt.gcf().suptitle(f"PSD (Averaged) Filtered - Subject {sub},"
                                   f" Run {run}")
                plt.show()

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
    if display:
        mne.viz.plot_events(
            events,
            sfreq=raw.info['sfreq'],
            first_samp=raw.first_samp,
            event_id=event_id
        )
    print(f"Dataset read: {len(epochs)} epochs")
    return epochs, event_id


def balanceClasses(epochs):
    event_id = epochs.event_id
    keys = list(event_id.keys())
    lenKey0 = len(epochs[keys[0]])
    lenKey1 = len(epochs[keys[1]])
    print(f"Balance Classes - Epochs for eventId {event_id[keys[0]]}: "
          f"{lenKey0}, eventId {event_id[keys[1]]}: {lenKey1}")
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
        print(f"Epochs for eventId {eventId} already have size "
              f"{len(event_indices)} ≤ target size {size}.")
        return epochs
    excess_indices = np.random.choice(
        event_indices,
        size=len(event_indices) - size, replace=False
    )
    print(f"Reducing size of eventId {eventId} from {len(event_indices)} "
          f"to {size} by dropping {len(excess_indices)} epochs.")

    epochs.drop(excess_indices)
    return epochs


def averageOverEpochs(epochs, event_id, epochsSizeForAverage=30):
    newEpochs = []
    newLabels = []

    keys = list(event_id.keys())

    epochsSize = min(len(epochs[keys[0]]), len(epochs[keys[1]]))

    print(f"Average over epochs - Epochs for eventId {event_id[keys[0]]}: "
          f"{len(epochs[keys[0]])}, eventId {event_id[keys[1]]}: "
          f"{len(epochs[keys[1]])}")

    i = 0
    while i < epochsSize:
        if i + epochsSizeForAverage > epochsSize:
            epochsSizeForAverage = epochsSize - i
        class0Avg = epochs[keys[0]][i:i+epochsSizeForAverage] \
            .average().get_data()
        class1Avg = epochs[keys[1]][i:i+epochsSizeForAverage] \
            .average().get_data()
        newEpochs.append(class0Avg)
        newLabels.append(event_id[keys[0]])
        newEpochs.append(class1Avg)
        newLabels.append(event_id[keys[1]])
        i += epochsSizeForAverage

    return np.array(newEpochs), np.array(newLabels)
