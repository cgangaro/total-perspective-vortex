import random
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from mne.decoding import CSP
from CSP import CSP
from WaveletFeatureExtractor import WaveletFeatureExtractor
from preprocess import PreProcessConfiguration, preprocess
from WaveletTransformer import WaveletTransformer


def main():

    config = PreProcessConfiguration(
        dataLocation="/home/cgangaro/sgoinfre/mne_data",
        loadData=False,
        saveData=True,
        makeMontage=True,
        montageShape="standard_1020",
        resample=True,
        resampleFreq=128.0,
        lowFilter=8.0,
        highFilter=32.0,
        ica=True,
        icaComponents=20,
        eog=True,
        epochsTmin=-1.,
        epochsTmax=3.
    )

    experiments = {
        0: [3, 7, 11],
        1: [4, 8, 12],
        2: [5, 9, 13],
        3: [6, 10, 14],
        4: [3, 4, 7, 8, 11, 12],
        5: [5, 6, 9, 10, 13, 14],
    }

    subjects1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    subjects2 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    subjects3 = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    subjects = subjects1 + subjects2 + subjects3
    # subjects = [1, 2, 3, 4]

    random.shuffle(subjects)
    sizeTestTab = int(len(subjects) * 0.2)
    testSubjects = subjects[:sizeTestTab]
    trainSubjects = subjects[sizeTestTab:]
    print(f"{len(trainSubjects)} train subjects, {len(testSubjects)} test subjects")
    
    dataTrainPreprocessed = preprocess(trainSubjects, experiments, config, "/home/cgangaro/sgoinfre/trainDataSave100")
    dataTestPreprocessed = preprocess(testSubjects, experiments, config, "/home/cgangaro/sgoinfre/testDataSave100")

    print("\n\n----------TRAIN DATA----------\n")
    print(f"Train data: {len(dataTrainPreprocessed)} experiments")
    models = {}
    for dataTrain in dataTrainPreprocessed:
        expId = dataTrain['experiment']
        epochs = dataTrain['epochs']
        labels = dataTrain['labels']
        subject_ids = dataTrain['subject_ids']
        
        epochs_data = epochs.get_data()

        print(f"Experiment {expId} - {epochs_data.shape} epochs, labels: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")

        cv = ShuffleSplit(
            n_splits=3,
            test_size=0.2
        )
        # cv = GroupKFold(n_splits=5)
        # cv = LeaveOneGroupOut()

        clf = make_pipeline(
            # CSP(n_components=6, reg=None, transform_into='csp_space', norm_trace=False),
            CSP(n_components=16, reg=None, log=True, norm_trace=False),
            # WaveletFeatureExtractor(wavelet='morl', scales=np.arange(1, 32), mode='magnitude'),
            StandardScaler(),
            # RandomForestClassifier(n_estimators=250, max_depth=None)
            LinearDiscriminantAnalysis(solver='svd', tol=0.0001)
        )
        scores = cross_val_score(clf, epochs_data, labels, cv=cv, groups=subject_ids, n_jobs=1)

        print(f"Experiment {expId} - Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores)*2:.2f})")

        clf.fit(epochs_data, labels)
        models[expId] = clf

    print("\n\n----------TEST DATA----------\n")
    print(f"Test data: {len(dataTestPreprocessed)} experiments, models: {len(models)}")

    accuracyTotal = 0
    for dataTest in dataTestPreprocessed:
        expId = dataTest['experiment']
        epochs = dataTest['epochs']
        labels = dataTest['labels']
        epochs_data = epochs.get_data()
        clf = models[expId]

        test_score = clf.score(epochs_data, labels)
        print(f"Experiment {expId} - Test Score: {test_score:.2f}")
        predictions = clf.predict(epochs_data)
        accuracy = accuracy_score(labels, predictions)
        print(f"Experiment {expId} - Test Accuracy: {accuracy:.2f}")
        accuracyTotal += accuracy
    accuracyTotal /= len(dataTestPreprocessed)
    print(f"Total Accuracy: {accuracyTotal:.4f}")
        
if __name__ == "__main__":
    main()


