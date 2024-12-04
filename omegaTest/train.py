import random
import numpy as np
import math
import mne
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from CSP import CSP
from MyCSP import MyCSP
# from utils.ExternalFilesProcess import saveModels
from sklearn.pipeline import Pipeline
# from utils.getArgsForTrain import getArgsForTrain

from nuageReadDataset import read_dataset
from read_dataset import read_dataset_batch, filter_raw, create_epochs

from assets import experiments

batch_read = 50
test_size = 0.2

subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109]


def main():

    try:
        print("Train program")
        
        expData = []
        for i, ex in enumerate(experiments):
            ex["epochs"] = []

            epochs, event_id = read_dataset(subjects, [3, 7, 11], {0: 'Rest', 1: 'Left fist', 2: 'Right fist'})

            epochs = balanceClasses(epochs)

            labels = epochs.events[:, -1]
            xTest, yTest, xTrain, yTrain = split_epochs_train_test(epochs, labels)
            xTrainAvg, yTrainAvg = averageOverEpochs(xTrain, event_id) 
            pipeline = make_clf()
            pipeline.fit(xTrainAvg, yTrainAvg)

            cv = ShuffleSplit(10, test_size=test_size, random_state=42)

            score = cross_val_score(
                estimator=pipeline,
                X=xTrainAvg,
                y=yTrainAvg,
                cv=cv,
                error_score='raise'
            )
            print(f"Crossval scores: {score}")
            expData.append({
                "experiment": i,
                "xTest": xTest,
                "yTest": yTest,
                "xTrain": xTrain,
                "yTrain": yTrain,
                "pipeline": pipeline,
                "score": score
            })

        print(f"TEST")
        test_scores = []
        train_scores = []
        crossval_scores = []
        print()

        for i, ex in enumerate(experiments):
            # X_test, y_test, X_train, y_train = split_train_test(ex)
            X_test = expData[i]["xTest"]
            y_test = expData[i]["yTest"]
            train_score = expData[i]["pipeline"].score(expData[i]["xTrain"], expData[i]["yTrain"])
            test_score = expData[i]["pipeline"].score(X_test, y_test)

            print(ex["name"])
            print("Train: ", train_score)
            train_scores.append(train_score)

            print("Test: ", test_score)
            test_scores.append(test_score)

            print("Crossval: %f" % (np.mean(expData[i]["score"])))
            crossval_scores.append(np.mean(expData[i]["score"]))

            print()

        print("Mean scores")
        print("Train: ", round(sum(train_scores) / len(train_scores), 2))
        print("Test: ", round(sum(test_scores) / len(test_scores), 2))
        print("Crossval: ", round(sum(crossval_scores) / len(crossval_scores), 2))
        return 
        

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

def split_epochs_train_test(E, y):
    """ **Creating data and targets**"""
    test_amount = math.ceil(test_size * len(E))

    E_test = E[:test_amount]
    y_test = y[:test_amount]

    E_train = E[test_amount:]
    y_train = y[test_amount:]

    return E_test, y_test, E_train, y_train

def averageOverEpochs(epochs, event_id, epochsSizeForAverage = 30):
    newEpochs = []
    newLabels = []

    keys = list(event_id.keys())
    print(f"keys: {keys}")

    epochsSize = min(len(epochs[keys[0]]), len(epochs[keys[1]]))

    i = 0
    while i < epochsSize:
        if i + epochsSizeForAverage > epochsSize:
            epochsSizeForAverage = epochsSize - i
        class0Avg = epochs[keys[0]][i:i+epochsSizeForAverage].average().get_data()
        class1Avg = epochs[keys[1]][i:i+epochsSizeForAverage].average().get_data()
        newEpochs.append(class0Avg)
        newLabels.append(event_id[keys[0]])
        newEpochs.append(class1Avg)
        newLabels.append(event_id[keys[1]])
        i += epochsSizeForAverage

    return np.array(newEpochs), np.array(newLabels)

def average_over_epochs(E_train, event_id):
    new_x = []
    new_y = []

    keys = list(event_id.keys())

    if len(E_train[keys[0]]) > len(E_train[keys[1]]):
        max_len = len(E_train[keys[1]])
    else:
        max_len = len(E_train[keys[0]])

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
            x_averaged = E_train[keys[0]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[0]])

            x_averaged = E_train[keys[1]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[1]])

            if i + avg_size >= len(E_train):
                avg_size = len(E_train) - i
            i = i + avg_size

    return np.array(new_x), np.array(new_y)

def balanceClasses(epochs):
    event_id = epochs.event_id
    keys = list(event_id.keys())
    lenKey0 = len(epochs[keys[0]])
    lenKey1 = len(epochs[keys[1]])

    if lenKey0 < lenKey1:
        epochs = setEpochsClassSize(epochs, event_id[keys[1]], lenKey0)
    elif lenKey1 < lenKey0:
        epochs = setEpochsClassSize(epochs, event_id[keys[0]], lenKey1)
    return epochs

def setEpochsClassSize(epochs, eventId, size):
    if len(epochs) < size:
        print("Not enough epochs to balance classes")
        return epochs
    event_indices = np.where(epochs.events[:, -1] == eventId)[0]
    if len(event_indices) <= size:
        print(f"Epochs for eventId {eventId} already have size {len(event_indices)} ≤ target size {size}.")
        return epochs
    excess_indices = np.random.choice(event_indices, size=len(event_indices) - size, replace=False)
    print(f"Reducing size of eventId {eventId} from {len(event_indices)} to {size} by dropping {len(excess_indices)} epochs.")

    epochs.drop(excess_indices)
    return epochs

def balance_classes(epochs):
    print("Balancing classes...")
    event_id = epochs.event_id
    keys = list(event_id.keys())
    print(f"Event id: {event_id}, keys: {keys}")
    exit()
    small, big = (keys[0], keys[1]) if len(epochs[keys[0]]) < len(
        epochs[keys[1]]) else (keys[1], keys[0])
    diff = len(epochs[big]) - len(epochs[small])

    indices = []
    for i in range(len(epochs.events[:, -1])):
        if len(indices) == diff:
            break
        if epochs.events[i, -1] == event_id[big]:
            indices.append(i)
    epochs.drop(indices)

    return epochs

def make_clf():
    csp = MyCSP(n_components=4)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')

    clf = Pipeline([
        ("CSP", csp),
        ("LDA", lda)
    ])
    return clf

def split_train_test(experiment):
    X = experiment["X"]
    y = experiment["y"]
    test_amount = math.ceil(test_size * len(X))

    X_test = X[:test_amount]
    y_test = y[:test_amount]

    X_train = X[test_amount:]
    y_train = y[test_amount:]

    return X_test, y_test, X_train, y_train

if __name__ == "__main__":
    main()
