import random
import numpy as np
import math
import mne
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.MyCSP import MyCSP
# from utils.ExternalFilesProcess import saveModels
from sklearn.pipeline import Pipeline
from utils.getArgsForTrain import getArgsForTrain

from nuagePreProcess import preprocessOneExperiment

shuffle_split_size = 0.2

def main():

    try:
        print("Train program")
        args = getArgsForTrain()
        subjects = args.datasetConfig.subjects
        experiments = args.experimentsConfig

        random.shuffle(subjects)
        sizeTestTab = int(len(subjects) * 0.2)
        testSubjects = subjects[:sizeTestTab]
        trainSubjects = subjects[sizeTestTab:]
        print(f"{len(trainSubjects)} train subjects, {len(testSubjects)} test subjects")

        print("\n\n----------TRAIN----------\n")
        expData = []
        for i, exp in enumerate(experiments):

            print(f"Experiment: {exp.id} - {exp.name}")
            print(f"Runs: {exp.runs}")
            print(f"Mapping: {exp.mapping}")
            
            xTrainAvg, yTrainAvg = preprocessOneExperiment(trainSubjects, exp.runs, exp.mapping, balance=True, average=True)
            xTest, yTest = preprocessOneExperiment(testSubjects, exp.runs, exp.mapping, balance=False, average=False)
            
            csp = MyCSP(n_components=4)
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')

            pipeline = Pipeline([
                ("CSP", csp),
                ("LDA", lda)
            ])

            pipeline.fit(xTrainAvg, yTrainAvg)

            cv = ShuffleSplit(10, test_size=0.2, random_state=42)

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
                "name": exp.name,
                "xTest": xTest,
                "yTest": yTest,
                "xTrain": xTrainAvg,
                "yTrain": yTrainAvg,
                "pipeline": pipeline,
                "score": score
            })

        print("\n\n----------TEST----------\n")
        print(f"Test data: {len(testSubjects)} subjects. Subjects: {testSubjects}\n")
        test_scores = []
        train_scores = []
        crossval_scores = []
        print()

        for i, exp in enumerate(experiments):
            data = expData[i]
            name = data["name"]
            X_test = data["xTest"]
            y_test = data["yTest"]
            train_score = data["pipeline"].score(data["xTrain"], data["yTrain"])
            test_score = data["pipeline"].score(X_test, y_test)

            print(f"Experiment {i} - {name}")
            print("Train: ", train_score)
            train_scores.append(train_score)

            print("Test: ", test_score)
            test_scores.append(test_score)

            print("Crossval: %f" % (np.mean(data["score"])))
            crossval_scores.append(np.mean(data["score"]))

            print()

        print("Mean scores")
        print("Train: ", round(sum(train_scores) / len(train_scores), 2))
        print("Test: ", round(sum(test_scores) / len(test_scores), 2))
        print("Crossval: ", round(sum(crossval_scores) / len(crossval_scores), 2))

        return 0

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
