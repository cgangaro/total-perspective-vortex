import random
import numpy as np
import math
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.CSP import CSP
from utils.MyCSP import MyCSP
from utils.preprocess import preprocess, preprocessMEGA
from utils.ExternalFilesProcess import saveModels
from utils.getArgsForTrain import getArgsForTrain

def main():

    try:
        print("Train program")
        
        args = getArgsForTrain()
        
        random.shuffle(args.subjects)
        sizeTestTab = int(len(args.subjects) * 0.25)
        testSubjects = args.subjects[:sizeTestTab]
        trainSubjects = args.subjects[sizeTestTab:]
        print(f"{len(trainSubjects)} train subjects, {len(testSubjects)} test subjects")
        
        # dataTrainPreprocessed = preprocess(
        #     saveDirectory=args.trainDataDir,
        #     config=args.preprocessConfig,
        #     subjects=trainSubjects,
        #     experiments=args.experimentsConfig,
        #     saveData=args.saveData,
        #     loadData=args.loadData
        # )
        # dataTestPreprocessed = preprocess(
        #     saveDirectory=args.testDataDir,
        #     config=args.preprocessConfig,
        #     subjects=testSubjects,
        #     experiments=args.experimentsConfig,
        #     saveData=args.saveData,
        #     loadData=args.loadData
        # )

        dataTrainPreprocessed, dataTestPreprocessed = preprocessMEGA(
            saveDirectory=args.trainDataDir,
            config=args.preprocessConfig,
            subjects=args.subjects,
            experiments=args.experimentsConfig,
            saveData=args.saveData,
            loadData=args.loadData
        )
        print(f"Train data: {dataTrainPreprocessed}")

        print("\n\n----------TRAIN DATA----------\n")
        print(f"Train data: {len(dataTrainPreprocessed)} experiments")
        models = {}
        for dataTrain in dataTrainPreprocessed:
            expId = dataTrain['experiment']
            epochs = dataTrain['epochs']
            labels = dataTrain['labels']
            event_id = dataTrain['event_id']
            # subject_ids = dataTrain['subject_ids']
            
            # epochs_data = epochs.get_data()
            epochs_data = epochs

            print(f"Experiment {expId} - {epochs_data.shape} epochs, labels: {labels.shape}")
            print(f"Unique labels: {np.unique(labels)}")
            epochs_data, labels = average_over_epochs(
                epochs_data,
                labels,
                event_id
            )

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
            # scores = cross_val_score(clf, epochs_data, labels, cv=cv, groups=subject_ids, n_jobs=1)
            scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

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
            # epochs_data = epochs.get_data()
            epochs_data = epochs
            clf = models[expId]

            test_score = clf.score(epochs_data, labels)
            print(f"Experiment {expId} - Test Score: {test_score:.2f}")
            predictions = clf.predict(epochs_data)
            accuracy = accuracy_score(labels, predictions)
            print(f"Experiment {expId} - Test Accuracy: {accuracy:.2f}")
            accuracyTotal += accuracy
        
        accuracyTotal /= len(dataTestPreprocessed)
        print(f"Total Accuracy: {accuracyTotal:.4f}")
        saveModels(models, args.modelsDir)


    except Exception as e:
        print("Error in train program: ", e)
        return 1


def average_over_epochs(X, y, event_id):
    print(f"X = {X}, y = {y}, event_id = {event_id}")
    # E_test, y_test, E_train, y_train = split_epochs_train_test(X, y)
    new_x = []
    new_y = []

    keys = list(event_id.keys())

    if len(X[keys[0]]) > len(X[keys[1]]):
        max_len = len(X[keys[1]])
    else:
        max_len = len(X[keys[0]])

    max_avg_size = 30
    min_amount_of_epochs = 5
    if max_len < min_amount_of_epochs * max_avg_size:
        max_avg_size = math.floor(max_len / min_amount_of_epochs)
    # Optional: averaging over multiple sizes to increase dataset size
    sizes = [max_avg_size]

    for avg_size in sizes:
        print("Averaging epochs over size: ", avg_size, "...")
        i = 0
        while i < max_len:
            x_averaged = X[keys[0]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[0]])

            x_averaged = X[keys[1]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[1]])

            if i + avg_size >= len(X):
                avg_size = len(X) - i
            i = i + avg_size

    return np.array(new_x), np.array(new_y)


if __name__ == "__main__":
    main()
