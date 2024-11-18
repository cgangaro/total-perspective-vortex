import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap, Xdawn)
from preprocessing.newPreprocessing import NewPreprocessing

def newPreprocess(subjects):
    print("newPreprocess")
    experiments = {
        0: [3, 7, 11],
        # 1: [4, 8, 12],
        # 2: [5, 9, 13],
        # 3: [6, 10, 14],
    }
    dataPreprocessed = []
    for expId in experiments:
        print(f"Preprocessing experiment {expId}")
        epochs, labels = getExperimentData(subjects, experiments[expId])
        dataPreprocessed.append(
            {
                'experiment': expId,
                'epochs': epochs,
                'labels': labels
            }
        )
    return dataPreprocessed

def getExperimentData(subjects, runs):
    print("getExperimentData, subjects: ", subjects, "runs: ", runs)
    raw_files = []
    for person in subjects:
        for task in runs:
            raw_execution = NewPreprocessing.loadRawFile(person, task)
            events, _ = events_from_annotations(raw_execution, event_id=dict(T0=1,T1=2,T2=3))
            mapping = {1:'rest', 2: 'do/feet', 3: 'do/hands'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
                orig_time=raw_execution.info['meas_date'])
            raw_execution.set_annotations(annot_from_events)
            raw_files.append(raw_execution)

    target_sfreq = 160
    for i in range(len(raw_files)):
        if raw_files[i].info['sfreq'] != target_sfreq:
            raw_files[i].resample(target_sfreq, npad="auto")
    raw = concatenate_raws(raw_files)
    print("End of raw concatenation")
    events, event_dict = events_from_annotations(raw)
    print(raw.info)
    print(event_dict)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    print(picks)
    biosemi_montage = make_standard_montage('biosemi64')
    # biosemi_montage.plot()
    # plt.show()
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    print("Montage set")
    # raw.plot(n_channels=3, duration = 2.5)
    # # plt.show()
    # raw.plot_psd(average=True)
    # # plt.show()
    # raw.plot_psd(average=False)
    # # plt.show()
    # fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
    # fig.subplots_adjust(right=0.7)  # make room for legend
    raw.filter(5., 40., fir_design='firwin', skip_by_annotation='edge')
    print("Filtering done")
    # raw.plot_psd(average=True)
    # plt.show()
    # raw.plot_psd(average=False)
    # plt.show()
    # raw_fastica = run_ica(raw, 'fastica', picks)
    # print("ICA done")
    # raw.plot(n_channels=25, start=0, duration=40,scalings=dict(eeg=250e-6))
    # raw_fastica.plot(n_channels=25, start=0, duration=40,scalings=dict(eeg=250e-6))
    # plt.show()
    event_id = {'do/feet': 1, 'do/hands': 2}
    tmin, tmax = -1., 4.

    events, event_dict = mne.events_from_annotations(raw, event_id=event_id)
    print(event_dict)

    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    print(epochs)
    print("Epochs data shape : ", epochs.get_data().shape)
    labels = epochs.events[:, -1] - 1
    print(labels)
    return epochs, labels

def run_ica(raw, method, picks, fit_params=None):
    raw_corrected = raw.copy()
    n_components=10
    
    ica = ICA(n_components=n_components, method=method, fit_params=fit_params, random_state=97)
    # t0 = time()
    ica.fit(raw_corrected, picks=picks)
    # fit_time = time() - t0
    # title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
    # ica.plot_components(title=title)
    # plt.show()
    
    eog_indices, scores = ica.find_bads_eog(raw, ch_name='Fpz',threshold=1.5)
    # ica.plot_scores(scores, exclude=eog_indices)  # look at r scores of components
    ica.exclude.extend(eog_indices) 
    raw_corrected = ica.apply(raw_corrected, n_pca_components = n_components, exclude = ica.exclude)
    print(ica.exclude, ica.labels_)
    return raw_corrected
