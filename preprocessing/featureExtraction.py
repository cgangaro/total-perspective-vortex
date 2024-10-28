import numpy as np
from mne import pick_types
from mne.preprocessing import ICA

class FeatureExtraction:
    @staticmethod
    def extractBandPower(epochs, bands):
        # Calculer la PSD pour chaque epoch et chaque canal
        psd, freqs = epochs.compute_psd().get_data(return_freqs=True)
        
        # Initialiser un dictionnaire pour stocker la puissance dans chaque bande de fréquence
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Sélectionner les indices des fréquences qui appartiennent à la bande de fréquence donnée
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            # Calculer la puissance dans cette bande pour chaque epoch et chaque canal
            band_power = np.sum(psd[:, :, freq_mask], axis=2)  # Intégrer sur les fréquences
            band_powers[band_name] = band_power  # Stocker la puissance calculée dans le dictionnaire
        
        return band_powers

    @staticmethod
    def ICAProcess(originRaw, n_components=20, useEOG=True):
        raw_corrected = originRaw.copy()
        # Ici n_components est le nombre de composantes indépendantes que l'ICA va extraire.
        # Nous avons 64 electrodes, donc nous pouvons en extraire 64 au maximum.
        # Mais en pratique, on en extrait moins, car certaines composantes sont du bruit.
        # Ici, on en extrait 20 car c'est un nombre communément utilisé.
        picks = pick_types(raw_corrected.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
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
        # ica.plot_components()
        if useEOG:
            eog_indicies, scores = ica.find_bads_eog(raw_corrected, ch_name='Fpz', threshold=1.5)
            # find_bads_eog permet de trouver les composantes ICA lies à des artefacts oculaires.
            # ch_name est le nom du canal EOG, ici Fpz correspond a une region proche des yeux, il capte donc fortement
            # les mouvements oculaires.
            # threshold est le seuil de detection des artefacts oculaires, ici on utilise 1.5.
            # eog_indicies contient les indices des composantes ICA correspondant à des artefacts oculaires.
            # scores contient les scores de correlation entre les composantes ICA et les artefacts oculaires (ici canal Fpz).
            print("eog_indicies: ", eog_indicies)
            # ica.plot_scores(scores, exclude=eog_indicies)
            ica.exclude.extend(eog_indicies)
            # On ajoute les indices des composantes ICA lies à des artefacts oculaires à la liste des composantes à exclure.
            raw_corrected = ica.apply(raw_corrected, n_pca_components = n_components, exclude = ica.exclude)
            # ica.apply permet de reconstruire notre signal EEG en excluant les composantes ICA lies à des artefacts oculaires.
        return raw_corrected
