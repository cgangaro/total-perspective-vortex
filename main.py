import mne
from mne.datasets import eegbci
import matplotlib
import matplotlib.pyplot as plt
from preprocessing.preprocessing import Preprocessing
from mne import pick_types
from mne.preprocessing import ICA 
matplotlib.use("webagg")


def main():
    try:
        subject = [1]
        run = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        montage_kind = "standard_1005"
        # montage_kind = "biosemi64"
        low_cutoff = 8.
        high_cutoff = 40.
        notch_freq = 60
        n_ica_components = 20

        original_raw, events, event_dict = Preprocessing.initRawFiles(subject, run)

        print("Make montage: ", montage_kind)
        original_raw = Preprocessing.makeMontage(original_raw, montage_kind, True)

        original_raw.plot(n_channels=5, scalings='auto', title='Données EEG brutes', show=True, block=True)
        Preprocessing.displayPSD(original_raw, "Donnees EEG brutes")

        raw = original_raw.copy()
        rawFiltered = Preprocessing.filterRawData(raw, low_cutoff, high_cutoff, notch_freq)
        rawFiltered.plot(n_channels=5, scalings='auto', title='Données EEG filtrées', show=True)
        Preprocessing.displayPSD(rawFiltered, "Donnees EEG filtrees")


        fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)

        tmin, tmax = -1., 2.
        epochs = mne.Epochs(rawFiltered, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=None, preload=True)
        labels = epochs.events[:, -1]

        epochs.plot(n_channels=5, title="Epochs filtrés")

        bands = {'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
        band_powers = FeatureExtraction.extract_band_power(epochs, bands)
        # band_powers.items() retourne un dictionnaire où chaque clé est une bande de fréquence (alpha, beta, gamma).
        # Chaque valeur est un tableau 2D où chaque ligne correspond à une epoch et chaque colonne à un canal EEG (electrode).

        for band, power in band_powers.items():
            print(f"{band} band power for each epoch:", power)

        # freqs = np.logspace(*np.log10([6, 30]), num=15)
        # wavelet_power = WaveletAnalysis.apply_wavelet_transform(epochs, freqs)

        # # Visualisation de l'analyse par ondelettes
        # wavelet_power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Analyse par ondelettes')

        return 
        raw_corrected = raw.copy()
        n_components = 20
        # Ici n_components est le nombre de composantes indépendantes que l'ICA va extraire.
        # Nous avons 64 electrodes, donc nous pouvons en extraire 64 au maximum.
        # Mais en pratique, on en extrait moins, car certaines composantes sont du bruit.
        # Ici, on en extrait 20 car c'est un nombre communément utilisé.
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        # pick_types permet de recuperer les canaux de notre analyse. Il permet de filtrer les canaux que l'on veut garder.
        # raw.info contient les informations sur les canaux de notre enregistrement, comme leur nom, leur type (EEG, EOG etc...),
        # leur frequence d'echantillonnage, etc...
        # meg permet de selectionner les canaux MEG (Magnétoencéphalographie), qui correspondent à des capteurs de champs magnétiques.
        # eeg permet de selectionner les canaux EEG, qui correspondent aux electrodes placées sur le cuir chevelu.
        # stim permet de selectionner les canaux de stimulation, qui sont utilises pour marquer les evenements.
        # Les evenements peuvent etre des stimuli visuels ou auditifs par exemple.
        # eog permet de selectionner les canaux EOG, qui permettent de detecter les mouvements oculaires.
        # exclude permet de selectionner les canaux à exclure, ici on exclut les canaux marqués comme mauvais.
        ica = ICA(n_components=n_components, method='fastica', fit_params=None, random_state=97)
        # ICA est une methode de séparation de signaux melanges en differentes composantes independantes.
        # Elle permet des d'isoler les signaux pertinents, et de supprimer les signaux parasites (comme les artefacts oculaires).
        # fastica est un algorithme d'ICA rapide et efficace.
        # fit_params est un dictionnaire de paramètres à passer à la méthode fit, ici on ne passe rien.
        # random_state est la seed d'initialisation de l'algorithme, pour garantir la reproductibilité des résultats.
        ica.fit(raw_corrected, picks=picks)
        # On entraine l'ICA sur nos données.
        ica.plot_components()
        eog_indicies, scores= ica.find_bads_eog(raw, ch_name='Fpz', threshold=1.5)
        # find_bads_eog permet de trouver les composantes ICA lies à des artefacts oculaires.
        # ch_name est le nom du canal EOG, ici Fpz correspond a une reginon proche des yeux, il capte donc fortement
        # les mouvements oculaires.
        # threshold est le seuil de detection des artefacts oculaires, ici on utilise 1.5.
        # eog_indicies contient les indices des composantes ICA correspondant à des artefacts oculaires.
        # scores contient les scores de correlation entre les composantes ICA et les artefacts oculaires (ici canal Fpz).
        print("eog_indicies: ", eog_indicies)
        ica.plot_scores(scores, exclude=eog_indicies)
        ica.exclude.extend(eog_indicies)
        # On ajoute les indices des composantes ICA lies à des artefacts oculaires à la liste des composantes à exclure.
        raw_corrected = ica.apply(raw_corrected, n_pca_components = n_components, exclude = ica.exclude)
        # ica.apply permet de reconstruire notre signal EEG en excluant les composantes ICA lies à des artefacts oculaires.

        # mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
        # raw.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
        # mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
        # plt.show()

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

        # subject = 1
        # runs = [3]  # Main gauche et droite

        # raw_fnames = eegbci.load_data(subject, runs)
        # print(raw_fnames)

        # raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])

        # # Configurer les informations des canaux pour les données EEG
        # raw.rename_channels(lambda x: x.strip('.'))

        # montage = mne.channels.make_standard_montage('standard_1020')
        # raw.set_montage(montage, on_missing='ignore')

        # Tracer les données brutes
        


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()