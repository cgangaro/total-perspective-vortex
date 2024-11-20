import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)})

import mne
from mne.decoding import LinearModel, Vectorizer, get_coef, Scaler, CSP, SPoC, UnsupervisedSpatialFilter
mne.set_log_level('WARNING')

import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import mne
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import StandardScaler
from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
import numpy as np
import mne
import random
from newPreprocess import newPreprocess, Configuration, PreProcessConfiguration
matplotlib.use("webagg")


def main():

    config = PreProcessConfiguration(
        withTargetSfreq=True,
        makeMontage=True,
        montageShape='standard_1020',
        lowFilter=8.0,
        highFilter=40.0,
        with128Hz=False,
        targetSfreq=160.0
    )

    subjects1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    subjects2 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    subjects3 = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    subjects = subjects1 + subjects2 + subjects3

    random.shuffle(subjects)
    sizeTestTab = int(len(subjects) * 0.25)
    testSubjects = subjects[:sizeTestTab]
    trainSubjects = subjects[sizeTestTab:]
    print(f"{len(trainSubjects)} train subjects, {len(testSubjects)} test subjects")
    dataTrainPreprocessed = newPreprocess(trainSubjects, config)
    # dataTestPreprocessed = newPreprocess(testSubjects, config)

    models = {}
    for data in dataTrainPreprocessed:
        print(data['epochs'])
        print(data['labels'])
        epochs = data['epochs']
        labels = data['labels']
        expId = data['experiment']
        score_pipeline = -1
        models_pipeline = None

        # Define a monte-carlo cross-validation generator (reduce variance):
        scores = []
        epochs_data = epochs.get_data()
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)

        # # Assemble a classifier
        # csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
        csp = SPoC(n_components=10)
        # rfc = RandomForestClassifier(n_estimators=200, random_state=42)
        rfc = LDA()
        print(f"Avant pipeline : epochs_data type: {epochs_data.dtype}, shape: {epochs_data.shape}")
        print(f"Labels type: {labels.dtype}, shape: {labels.shape}")
        print(f"epochs_data type: {epochs_data.dtype}, shape: {epochs_data.shape}")
        # Use scikit-learn Pipeline with cross_val_score function
        clf = make_pipeline(csp, rfc)
        scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

        # Printing the results
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        clf.fit(epochs_data, labels)
        models[expId] = clf

    experiment_accuracies = {}

    # for dataTest in dataTestPreprocessed:
    #     epochs = dataTest['epochs']
    #     labels = dataTest['labels']
    #     expId = dataTest['experiment']
    #     epochs_data = epochs.get_data()
    #     print(f"epochs_data type: {epochs_data.dtype}, shape: {epochs_data.shape}")
    #     print(f"Labels type: {labels.dtype}, shape: {labels.shape}")
    #     clf = models[expId]
    #     score = clf.score(epochs_data, labels)
    #     print(f"Score for experiment {expId}: {score}")
    #     predictions = clf.predict(epochs_data)
    #     accuracy = accuracy_score(labels, predictions)
    #     print(f"Test accuracy for experiment {expId}: {accuracy:.4f}")
    #     experiment_accuracies[expId] = accuracy

    all_subject_scores = {}
    print(f"----------Test subjects: {testSubjects}----------")
    for subject in testSubjects:
        print(f"----------Subject {subject} has {len(testSubjects)} experiments----------")
        dataTestPreprocessed = newPreprocess([subject], config)

        subject_results = {}
        for dataTest in dataTestPreprocessed:
            epochs = dataTest['epochs']
            labels = dataTest['labels']
            expId = dataTest['experiment']
            epochs_data = epochs.get_data()
            # print(f"Subject {subject} experiment {expId}, epochs_data shape: {epochs_data.shape}, labels shape: {labels.shape}, labels: {labels}")
            clf = models[expId]
            score = clf.score(epochs_data, labels)
            predictions = clf.predict(epochs_data)
            accuracy = accuracy_score(labels, predictions)
            # print(f"Score : {score}, Accuracy: {accuracy}")
            subject_results[expId] = {
                'score': score,
                'accuracy': accuracy
            }

        all_subject_scores[subject] = subject_results

    print()
    print()
    print("----------Results----------")
    print()
    for subject, subject_results in all_subject_scores.items():
        for expId, results in subject_results.items():
            if expId not in experiment_accuracies:
                experiment_accuracies[expId] = []
            experiment_accuracies[expId].append(results['accuracy'])

    for expId, accuracies in experiment_accuracies.items():
        mean_accuracy = np.mean(accuracies)
        print(f"List of accuracies for experiment {expId}: {accuracies}")
        print(f"Expérience {expId} : Précision moyenne = {mean_accuracy:.4f} sur {len(accuracies)} sujets")
    # if experiment_accuracies:
    #     mean_accuracy = np.mean(list(experiment_accuracies.values()))
    #     print(f"Mean test accuracy across all experiments: {mean_accuracy:.4f}")
    # else:
    #     print("No experiments were evaluated.")
    return

if __name__ == "__main__":
    main()
