import joblib
import matplotlib
import mne
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from cspTransformer import CSPTransformer
from preProcess import preProcess, Configuration, PreProcessConfiguration, splitData, newSplitData
from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
matplotlib.use("webagg")

def main():
    print("LAUNCHING MAIN")
    preProcessConfig = PreProcessConfiguration(
        ICA=True,
        EOG=True,
        montageKind="standard_1005",
        lowCutoff=5.,
        highCutoff=40.,
        notchFreq=60,
        nIcaComponents=20,
        useRestMoment=False
    )
    config = Configuration(
        preProcess=preProcessConfig,
        CSPTransformerNComponents=4,
        dataDir='data',
        saveToFile=True,
        loadFromFile=False
    )
    # subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    subjects1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    subjects2 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    subjects3 = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    # subjects = subjects1 + subjects2 + subjects3
    subjects = [1, 2, 3]
    # experiments = {
    #     0: [3, 7, 11],
    #     1: [],
    #     2: [],
    #     3: [],
    # }
    #     experiments = {
    #     0: [3, 7, 11],
    #     1: [4, 8, 12],
    #     2: [5, 9, 13],
    #     3: [6, 10, 14],
    # }
    experiments = {
        0: [],
        1: [],
        2: [5, 9, 13],
        3: [6, 10, 14],
    }


    dataLoaded = preProcess(
        subjects=subjects,
        experiments=experiments,
        config=config,
        save_to_file=config.saveToFile,
        output_dir=config.dataDir,
        name='none'
    )

    # dataSplit = splitData(dataLoaded, experiments)
    dataSplit = newSplitData(dataLoaded, experiments)

    # Entraîner le modèle
    transformer = CSP(n_components=10, reg=None, log=True, norm_trace=False)
    scaler = StandardScaler()
    classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    # pipeline = Pipeline([
    #     ('CSP', transformer),
    #     ('RandomForestClassifier', classifier)
    # ])
    pipeline = Pipeline([
        ('CSP', CSP(n_components=10)),
        ('StandardScaler', StandardScaler()),
        ('RandomForestClassifier', RandomForestClassifier(n_estimators=150))
    ])

    print("EVALUATING MODELS")
    expAccuraciesValidation = {}
    expAccuraciesTest = {}
    for expId in experiments:
        try:
            print(f"Expérience {expId}")
            # pipelineFit = getModelFit(dataSplit[expId]['X_train'], dataSplit[expId]['labels_train'], pipeline, "d'entraînement", expId)
            # validationAccuracy = getAccuracyScore(dataSplit[expId]['X_validation'], dataSplit[expId]['labels_validation'], pipelineFit, "de validation", expId)
            # testAccuracy = getAccuracyScore(dataSplit[expId]['X_test'], dataSplit[expId]['labels_test'], pipelineFit, "de test", expId)
            # pipeline = make_pipeline(CSP(n_components=10), StandardScaler(), RandomForestClassifier(n_estimators=150))
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            # print("X type: ", type(dataSplit[expId]['X']))
            # print("labels type: ", type(dataSplit[expId]['labels']))
            X = dataSplit[expId]['X']
            labels = dataSplit[expId]['labels']
            csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
            rfc = RandomForestClassifier(n_estimators=150, random_state=42)
            clf = make_pipeline(csp, rfc)
            print(f"Avant pipeline : X type: {X.dtype}, shape: {X.shape}, labels type: {labels.dtype}, shape: {labels.shape}")
            scores = cross_val_score(clf, X, labels, cv=cv, n_jobs=-1)
            print(f"Précision moyenne : {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
            # print(f"############################################# Validation Accuracy: {validationAccuracy:.4f}, Test Accuracy: {testAccuracy:.4f}")
            # expAccuraciesValidation[expId] = validationAccuracy
            # expAccuraciesTest[expId] = testAccuracy
        except ValueError as e:
            print(e)
            expAccuraciesValidation[expId] = 0
            expAccuraciesTest[expId] = 0
            continue
        except Exception as e:
            print(f"Erreur lors de l'évaluation du modèle pour l'expérience {expId}: {e}")
            continue
    
    print("FIN DE L'ÉVALUATION")
    testAccuraciesAverage = np.mean(list(expAccuraciesTest.values()))
    validationAccuraciesAverage = np.mean(list(expAccuraciesValidation.values()))
    for expId in experiments:
        print(f"Expérience {expId} - Précision de validation: {expAccuraciesValidation[expId]:.4f}, Précision de test: {expAccuraciesTest[expId]:.4f}")
    print(f"Précision moyenne de validation: {validationAccuraciesAverage:.4f}, Précision moyenne de test: {testAccuraciesAverage:.4f}")

if __name__ == "__main__":
    main()
