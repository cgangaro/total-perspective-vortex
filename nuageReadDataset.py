import mne
import logging
from config import DATASET_PATH
from mne import pick_types
mne.set_log_level('CRITICAL')
mne.set_log_level('WARNING')
mne.set_log_level('ERROR') 
logging.basicConfig(level=logging.WARNING)

good_channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]

# Ex:
# mapping = {0: 'Rest', 1: 'Left fist', 2: 'Right fist'}
def read_dataset(subjects, runs, mapping):
    raws = []

    for sub in subjects:
        for run in runs:
            filePath = DATASET_PATH + f"S{sub:03d}/S{sub:03d}R{run:02d}.edf"
            print(f"Reading file: {filePath}")
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