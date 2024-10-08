import mne
from mne.datasets import eegbci
import matplotlib
matplotlib.use("webagg")
import matplotlib.pyplot as plt
from preprocessing.preprocessing import Preprocessing

def main():
    try:
        print("Hello, World!")
        raw = Preprocessing.initRawFiles([1])
        Preprocessing.testTest()

        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)

        # subject = 1
        # runs = [3]  # Main gauche et droite

        # raw_fnames = eegbci.load_data(subject, runs)
        # print(raw_fnames)

        # raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])

        # # Configurer les informations des canaux pour les données EEG
        # raw.rename_channels(lambda x: x.strip('.'))

        # montage = mne.channels.make_standard_montage('standard_1020')
        # raw.set_montage(montage, on_missing='ignore')

        # # Tracer les données brutes
        # raw.plot(n_channels=10, scalings='auto', title='Données EEG brutes', show=True, block=True)
        # raw.plot_psd(average=True)

        # raw.filter(l_freq=8., h_freq=40., fir_design='firwin', skip_by_annotation='edge')

        # raw.plot(n_channels=10, scalings='auto', title='Données EEG filtrées (1-40 Hz)', show=True)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()