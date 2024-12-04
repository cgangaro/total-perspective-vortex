from config import DATASET_PATH
from config import good_channels
from config import experiments, amount_of_subjects, amount_of_runs

import mne
import sys
from mne import pick_types


def get_filepath(subject_nr=1, run_nr=1):
    subject = "S{:03d}".format(subject_nr)
    run = "R{:02d}".format(run_nr)
    filename = subject + "/" + subject + run + ".edf"
    filepath = DATASET_PATH + filename
    return filepath


def make_runs():
    """**Creating a metadata array containing information about all runs**"""
    runs = []
    for subject_nr in range(1, amount_of_subjects + 1):
        for run_nr in range(3, amount_of_runs + 1):
            filepath = get_filepath(subject_nr=subject_nr, run_nr=run_nr)
            runs.append([run_nr, filepath])
    return runs


def get_mapping(run_nr):
    event_mapping = {}
    for e in experiments:
        if run_nr in e["runs"]:
            event_mapping = e["mapping"]
    return event_mapping


def filter_raw(raw):
    """**Filtering**

    - Simple bandpass
    - Notch filter to filter out 60hz electrical signals
    """
    f_low = 1.0
    f_high = 15.0

    raw_filtered = raw.copy()
    raw_filtered.notch_filter(60, method="iir")
    raw_filtered.filter(f_low, f_high, fir_design="firwin",
                        skip_by_annotation="edge")
    return raw_filtered


def read_dataset_batch(ex_nr, batch, start, runs=make_runs()):
    print("Reading dataset batch")
    print(f"ex_nr: {ex_nr}")
    # print(f"run: {runs}")
    raws = []

    batch_counter = batch
    start_counter = start
    for r in runs:
        print(f"r: {r}")
        if r[0] in experiments[ex_nr]["runs"]:
            print(f"r[0]: {r[0]}")
            print(f"experiments[ex_nr]['runs']: {experiments[ex_nr]['runs']}")
            if batch_counter == 0:
                break
            if start_counter == 0:
                raw = mne.io.read_raw_edf(r[1], preload=True)
                if raw.info['sfreq'] != 160.0:
                    raw.resample(sfreq=160.0)
                mne.datasets.eegbci.standardize(raw)
                raw.set_montage("standard_1005")

                events, _ = mne.events_from_annotations(
                    raw,
                    event_id=dict(T1=1, T2=2))
                mapping = get_mapping(r[0])
                print(f"mapping: {mapping}")
                return
                annotations = mne.annotations_from_events(
                    events=events,
                    event_desc=mapping,
                    sfreq=raw.info["sfreq"]
                )
                raw.set_annotations(annotations)
                raws.append(raw)
                batch_counter -= 1
            else:
                start_counter -= 1

    if len(raws) == 0:
        return None
    raw = mne.concatenate_raws(raws)

    channels = raw.info["ch_names"]
    bad_channels = [x for x in channels if x not in good_channels]
    raw.drop_channels(bad_channels)

    return raw


def read_subject(subject, run):
    print("Reading subject", subject, "run", run)
    path = get_filepath(subject, run)
    raw = mne.io.read_raw_edf(path, preload=True)
    if raw.info['sfreq'] != 160.0:
        raw.resample(sfreq=160.0)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage("standard_1005")

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=dict(T1=1, T2=2))
    annotations = mne.annotations_from_events(
        events=events,
        sfreq=raw.info["sfreq"]
    )
    raw.set_annotations(annotations)

    raw_filtered = filter_raw(raw)

    channels = raw_filtered.info["ch_names"]
    bad_channels = [x for x in channels if x not in good_channels]
    raw_filtered.drop_channels(bad_channels)

    epochs = create_epochs(raw_filtered, events, event_id)
    return raw, epochs


def create_epochs(raw, events, event_id):
    """**Creating epochs**

    In the MNE-Python library, an "epoch" is a defined time window of EEG
    (Electroencephalography) or MEG (Magnetoencephalography) data that is
    extracted from continuous data based on specific events or triggers.
    """
    tmin = -.500  # start of each epoch (in sec)
    tmax = 1.000  # end of each epoch (in sec)
    baseline = (None, 0)
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")

    epochs = mne.Epochs(raw,
                        events=events,
                        event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=baseline,
                        picks=picks,
                        proj=True,
                        preload=True)

    return epochs


def check_subject_and_run(subject, run):
    if subject > 109 or subject < 1:
        print("Subject number must be any value from 1 to 109")
        sys.exit(1)
    if run > 14 or run < 3:
        print(
            "Run number must be any value from 3 to 14 (1 and 2 are baseline)"
        )
        sys.exit(1)
