import mne
from mne.datasets import eegbci
import matplotlib
<<<<<<< HEAD
import matplotlib.pyplot as plt
from preprocessing.preprocessing import Preprocessing
from mne.channels import make_standard_montage
matplotlib.use("webagg")


def main():
    try:
        raw = Preprocessing.initRawFiles([1])
        # Maybe filter bad eog
        rawFiltered = raw.copy().filter(l_freq=8., h_freq=40., fir_design='firwin', skip_by_annotation='edge')
        # Preprocessing.visualizeRawData(raw, rawFiltered, 5, "8-40Hz")
        eegbci.standardize(raw)
        montage = make_standard_montage("biosemi64")
        raw.set_montage(montage, on_missing='ignore')
        montage = raw.get_montage()
        montage.plot()
        mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
        raw.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
        mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
        plt.show()

        # # plot
        # if plotIt:
        #     montage = raw.get_montage()
        #     p = montage.plot()
        #     p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})

        # raw.plot(n_channels=10, scalings='auto', title='Données EEG brutes', show=True, block=True)
        # raw.plot(n_channels=10, scalings='auto', title='Données EEG filtrées (1-40 Hz)', show=True)

        # eegbci.standardize(raw)
        # print("Visualisation des données brutes avant filtrage...")
        # raw.plot(n_channels=10, scalings='auto', title='Données EEG brutes', show=True, block=True)

        # print("Application du filtre passe-bande (8-30 Hz)...")
        # raw_filtered = raw.copy().filter(l_freq=8., h_freq=30., fir_design='firwin')
        
        # # Étape 3 : Visualisation des données après filtrage
        # print("Visualisation des données après filtrage...")
        # raw_filtered.plot(title="Données filtrées (8-30 Hz)", block=True)

        #         raw.plot_psd(average=True)

        # raw.filter(l_freq=8., h_freq=40., fir_design='firwin', skip_by_annotation='edge')

        # raw.plot(n_channels=10, scalings='auto', title='Données EEG filtrées (1-40 Hz)', show=True)
        # montage = mne.channels.make_standard_montage('standard_1005')
        # raw.set_montage(montage)
=======
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
>>>>>>> ee641d884eb906081ebd20c3f15b5191d9530846

        # subject = 1
        # runs = [3]  # Main gauche et droite

        # raw_fnames = eegbci.load_data(subject, runs)
        # print(raw_fnames)

        # raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])

        # # Configurer les informations des canaux pour les données EEG
        # raw.rename_channels(lambda x: x.strip('.'))

        # montage = mne.channels.make_standard_montage('standard_1020')
        # raw.set_montage(montage, on_missing='ignore')

<<<<<<< HEAD
        # Tracer les données brutes
        

=======
        # # Tracer les données brutes
        # raw.plot(n_channels=10, scalings='auto', title='Données EEG brutes', show=True, block=True)
        # raw.plot_psd(average=True)

        # raw.filter(l_freq=8., h_freq=40., fir_design='firwin', skip_by_annotation='edge')

        # raw.plot(n_channels=10, scalings='auto', title='Données EEG filtrées (1-40 Hz)', show=True)
>>>>>>> ee641d884eb906081ebd20c3f15b5191d9530846

    except Exception as e:
        print(f"An error occurred: {e}")

<<<<<<< HEAD

=======
>>>>>>> ee641d884eb906081ebd20c3f15b5191d9530846
if __name__ == "__main__":
    main()