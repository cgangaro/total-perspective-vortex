import mne
import matplotlib.pyplot as plt

from config import plotting
from config import csp_params
from MyCSP import MyCSP


def plot_raw(raw):
    """**Plot an instance of raw data**"""
    if plotting:
        events, _ = mne.events_from_annotations(raw)
        raw.plot(scalings=dict(eeg=250e-6), events=events)


def plot_psd(raw):
    """**Plotting the PSD of the raw data**"""
    if plotting:
        raw.plot_psd()


def plot_evoked(epochs):
    """**Plotting an evoked**"""
    if plotting:
        keys = list(epochs.event_id.keys())

        evoked = epochs[keys[0]].average()

        evoked.plot(gfp=True)
        evoked.plot_topomap(times=[-0.2, 0.2, 0.4, 0.6, 0.8], average=0.05)


def plot_epochs_image(epochs):
    if plotting:
        keys = list(epochs.event_id.keys())

        epochs[keys[0]].plot_image(picks=["Cz"])


def plot_csp_separation(X, y):
    """Plot a scatter_plot of X_train after CSP transformation"""
    if plotting:
        csp = MyCSP(csp_params["n_components"])
        X = csp.fit_transform(X, y)

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()


def plot_compare_evokeds(epochs):
    if plotting:
        keys = list(epochs.event_id.keys())
        mne.viz.plot_compare_evokeds(
            dict(left_fist=epochs[keys[0]].average(),
                 right_fist=epochs[keys[1]].average()),
            legend="upper left"
        )
