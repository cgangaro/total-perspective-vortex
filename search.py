import joblib
import matplotlib
import mne
import os
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from cspTransformer import CSPTransformer
from preProcess import preProcess, Configuration, PreProcessConfiguration
from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
matplotlib.use("webagg")


def main():
    print("LAUNCHING MAIN")
    # Charger les données prétraitées
    # montageKind="standard_1005",
    preProcessConfig = PreProcessConfiguration(
        ICA=True,
        EOG=True,
        montageKind="standard_1020",
        lowCutoff=8.,
        highCutoff=32.,
        notchFreq=60,
        nIcaComponents=20,
        useRestMoment=False
    )
    config = Configuration(
        preProcess=preProcessConfig,
        CSPTransformerNComponents=4,
        dataDir='data',
        saveToFile=True,
        loadFromFile=True
    )
    # subjects = [1]
    subjects1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    subjects2 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    subjects3 = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    subjects = subjects1 + subjects2
    # experiments = {
    #     0: [1, 2],     # Baseline
    #     1: [3, 7, 11], # Mouvements main gauche
    #     2: [4, 8, 12], # Mouvements main droite
    #     3: [5, 9, 13], # Mouvements des pieds
    #     4: [6, 10, 14],# Mouvements de la langue
    #     5: [15, 16]    # Autres expériences si applicables
    # }
    experiments = {
        0: [],     # Baseline
        1: [], # Mouvements main gauche
        2: [4, 8, 12], # Mouvements main droite
        3: [], # Mouvements des pieds
        4: [],# Mouvements de la langue
        5: []    # Autres expériences si applicables
    }

    experimentData = {}
    if config.loadFromFile:
        print("LOADING DATA FROM FILE")
        for exp_id in experiments:
            nbSubjects = len(subjects)
            outputPath = os.path.join(config.dataDir, f"experiment_{exp_id}_n{nbSubjects}.npz")
            if not os.path.exists(outputPath):
                print(f"Le fichier {outputPath} n'existe pas.")
                continue
            loaded = np.load(outputPath, allow_pickle=True)
            itemLoaded = loaded['arr_0'].item()
            experimentData[exp_id] = itemLoaded
    else:
        print("PREPROCESSING DATA")
        experimentData = preProcess(
            subjects=subjects,
            experiments=experiments,
            config=config,
            save_to_file=config.saveToFile,
            name='experiment1'
        )
    print("DATA LOADED")
    if True:
        for exp_id, data in experimentData.items():
            if len(data['X']) == 0:
                print(f"Aucune donnée n'est disponible pour l'expérience {exp_id}")
                continue
            X = data['X']
            labels = data['labels']
            print(f"Expérience {exp_id} - X shape: {X.shape}, labels shape: {labels.shape} - T1 count: {np.sum(labels == 0)}, T2 count: {np.sum(labels == 1)}")

    # for exp_id, data in data_per_experiment.items():
    #     if exp_id == 3:
    #         print(data)
    #         print("Data shape: ", data['X'].shape)

    # {np.str_('do/feet'): 1, np.str_('do/hands'): 2}
    # <Epochs | 45 events (all good), -1 – 4 s (baseline off), ~17.7 MB, data loaded,
    #  'do/feet': 23
    #  'do/hands': 22>
    # Epochs data shape :  (45, 64, 801)
    # [1 0 1 0 1 0 0 1 1 0 1 0 0 1 0 0 1 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 0 1 1 0 1
    #  0 0 1 0 1 1 0 1]
    
    transformers = {
        'CSP': CSP(n_components=10, reg=None, log=True, norm_trace=False),
        # 'SPoC': SPoC(n_components=10, log=True, reg='oas', rank='full')
    }

    classifiers = {
        'LDA': LDA(),
        # 'RandomForest': RandomForestClassifier(n_estimators=150, random_state=42)
    }

    cv = ShuffleSplit(n_splits=8, test_size=0.3, random_state=42)

    all_exp_scores = []
    generalScores = []
    print("EVALUATING MODELS")
    for exp_id, data in experimentData.items():
        print(f"Expérience {exp_id}")
        if data is None:
            print(f"Expérience {exp_id} ignorée car aucune donnée n'est disponible.")
            continue
        X = data['X'].astype(np.float64)
        labels = data['labels']
        # groups = data['groups']
        expScores = []
        for transformer_name, transformer in transformers.items():
            for classifier_name, classifier in classifiers.items():
                pipeline = Pipeline([
                    (transformer_name, transformer),
                    (classifier_name, classifier)
                ])

                # Validation croisée LeaveOneGroupOut
                # logo = LeaveOneGroupOut()
                scores = cross_val_score(pipeline, X, labels, cv=cv, scoring='accuracy', n_jobs=8)
                mean_score = scores.mean()
                all_exp_scores.append(mean_score)
                print(f"Expérience {exp_id} - {transformer_name} + {classifier_name}: précision moyenne = {mean_score:.4f}")
                expScores.append(f"Expérience {exp_id} - {transformer_name} + {classifier_name}: précision moyenne = {mean_score:.4f}\n")
                # Entraîner le modèle sur l'ensemble des données
                # pipeline.fit(X, labels)
                # # Sauvegarder le modèle
                # model_filename = f'model_exp_{exp_id}_{transformer_name}_{classifier_name}.pkl'
                # joblib.dump(pipeline, model_filename)
                # print(f"Modèle pour l'expérience {exp_id} avec {transformer_name} + {classifier_name} sauvegardé dans {model_filename}")
        generalScores.append(expScores)
    if all_exp_scores:
        overall_mean_accuracy = np.mean(all_exp_scores)
        print(f"Précision moyenne sur toutes les expériences: {overall_mean_accuracy:.4f}")
    else:
        print("Aucune expérience n'a pu être évaluée.")
    for exp in generalScores:
        print(exp)

if __name__ == "__main__":
    main()
