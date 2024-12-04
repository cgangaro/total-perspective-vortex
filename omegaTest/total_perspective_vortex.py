import joblib
import math
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import mne
import sys
import argparse

from MyCSP import MyCSP
from config import experiments, batch_read
from config import csp_params
from config import test_size, amount_of_subjects

from read_dataset import read_dataset_batch
from read_dataset import create_epochs
from read_dataset import read_subject
from read_dataset import check_subject_and_run
from read_dataset import filter_raw

from plotting import plot_raw
from plotting import plot_psd
from plotting import plot_evoked
from plotting import plot_epochs_image
from plotting import plot_csp_separation
from plotting import plot_compare_evokeds


"""
1. Left or right fist [R3, R7, R11]
2. Imagine left or right fist [R4, R8, R12]
3. Both fists or feet [R5, R9, R13]
4. Imagine both fists or feet [R6, R10, R14]
5. Baseline eyes open [R1]
6. Baseline eyes closed [R2]

# V.1.1 Preprocessing, parsing and formatting

First, you’ll need to parse and explore EEG data with MNE, from physionet.
You will have to write a script to visualize raw data and then filter it to
keep only useful frequency bands, and visualize again after this preprocessing.

This part is where you’ll decide which features you’ll extract from the
signals to feed them to your algorithm. So you’ll have to be thorough in
picking what matters for the desired output.

One example is to use the power of the signal by frequency and by channel to
the pipeline’s input. Most of the algorithms linked to filtering and obtaining
the signal’s specter use Fourier transform or wavelet transform (cf. bonus).

**Experiment structure**

T0 corresponds to rest

T1 corresponds to onset of motion (real or imagined) of:
  - the left fist (in runs 3, 4, 7, 8, 11, and 12)
  - both fists (in runs 5, 6, 9, 10, 13, and 14)

T2 corresponds to onset of motion (real or imagined) of:
  - the right fist (in runs 3, 4, 7, 8, 11, and 12)
  - both feet (in runs 5, 6, 9, 10, 13, and 14)

Therefore there are 5 targets:
  - Rest
  - Left fist
  - Right fist
  - Both fists
  - Both feet

**DATASET STRUCTURE:**

```
dataset/S{subject_nr}/S{subject_nr}{run_nr}.edf
```
"""


def balance_classes(epochs):
    print("Balancing classes...")
    event_id = epochs.event_id
    keys = list(event_id.keys())

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


def split_epochs_train_test(E, y):
    """ **Creating data and targets**"""
    test_amount = math.ceil(test_size * len(E))

    E_test = E[:test_amount]
    y_test = y[:test_amount]

    E_train = E[test_amount:]
    y_train = y[test_amount:]

    return E_test, y_test, E_train, y_train


def average_over_epochs(X, y, event_id):
    E_test, y_test, E_train, y_train = split_epochs_train_test(X, y)
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


"""# V.1.2 Treatment pipeline

Then the processing pipeline has to be set up:

• Dimensionality reduction algorithm (i.e.: PCA, ICA, CSP, CSSP...)

• Classification algorithm, there is plenty of choice among those available in
sklearn, to output the decision of what data chunk corresponds to what kind of
motion.

• "Playback" reading on the file to simulate a data stream.

It is advised to first test your program architecture with sklearn and MNE
algorithms, before implementing your own CSP or whatever algorithm you chose.

The program will have to contain a script for training and a script for
prediction.
The script predicting output will have to do it on a stream of data, and
within a delay of 2s after the data chunk was sent to the processing pipeline.
(You should not use mne-realtime)

You have to use the pipeline object from sklearn (use baseEstimator and
transformerMixin classes of sklearn)


# V.1.4 Train, Validation, and Test

• You have to use cross_val_score on the whole processing pipeline, to
evaluate your classification.

• You must choose how to split your data set between Train, Validation, and
Test set (Do not overfit, with different splits each time)

• You must have 60% mean accuracy on all subjects used in your Test Data
(corresponding to the six types of experiment runs and on never-learned data)

• You can train/predict on the subject and the task of your choice
"""


def make_clf():
    csp = MyCSP(csp_params["n_components"])
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')

    clf = Pipeline([
        ("CSP", csp),
        ("LDA", lda)
    ])
    return clf


def dump_model(clf, name, amount_of_subjects):
    joblib.dump(
        clf,
        f'{name}_{amount_of_subjects}_subjects.joblib'
    )


def split_train_test(experiment):
    X = experiment["X"]
    y = experiment["y"]
    test_amount = math.ceil(test_size * len(X))

    X_test = X[:test_amount]
    y_test = y[:test_amount]

    X_train = X[test_amount:]
    y_train = y[test_amount:]

    return X_test, y_test, X_train, y_train


def score_and_print_results():
    test_scores = []
    train_scores = []
    crossval_scores = []
    print()

    for ex in experiments:
        X_test, y_test, X_train, y_train = split_train_test(ex)
        train_score = ex["clf"].score(ex["X_avg"], ex["y_avg"])
        test_score = ex["clf"].score(X_test, y_test)

        print(ex["name"])
        print("Train: ", train_score)
        train_scores.append(train_score)

        print("Test: ", test_score)
        test_scores.append(test_score)

        print("Crossval: %f" % (np.mean(ex["crossval_scores"])))
        crossval_scores.append(np.mean(ex["crossval_scores"]))

        print()

    print("Mean scores")
    print("Train: ", round(sum(train_scores) / len(train_scores), 2))
    print("Test: ", round(sum(test_scores) / len(test_scores), 2))
    print("Crossval: ", round(sum(crossval_scores) / len(crossval_scores), 2))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict the classes of a given EEG signal"
    )
    parser.add_argument("subject", type=int, help="Subject number")
    parser.add_argument("run", type=int, help="Run number")
    args = parser.parse_args()

    check_subject_and_run(args.subject, args.run)
    return args


def main():
    """
    If no arguments are given, train and test the model on all subjects.
    Otherwise, train and test the model on the given subject and run.
    """
    cv = ShuffleSplit(10, test_size=test_size, random_state=42)

    if len(sys.argv) < 3:
        for i, ex in enumerate(experiments):
            ex["epochs"] = []
            batch_start = 0
            buffer = read_dataset_batch(i, batch_read, batch_start)

            # Plot raw
            plot_raw(buffer)
            plot_psd(buffer)
            buffer_filtered = filter_raw(buffer)
            # Plot filtered raw
            plot_psd(buffer_filtered)

            while buffer is not None:
                buffer = filter_raw(buffer)
                ex["raw"] = buffer
                events, event_id = mne.events_from_annotations(ex["raw"])
                batch_start += batch_read

                ex["epochs"].append(create_epochs(ex["raw"], events, event_id))
                del ex["raw"]
                buffer = read_dataset_batch(i, batch_read, batch_start)

            ex["epochs"] = mne.concatenate_epochs(ex["epochs"])
            ex["epochs"] = balance_classes(ex["epochs"])
            plot_evoked(ex["epochs"])
            plot_epochs_image(ex["epochs"])
            plot_compare_evokeds(ex["epochs"])

            ex['y'] = ex["epochs"].events[:, -1]

            ex["X_avg"], ex["y_avg"] = average_over_epochs(
                ex["epochs"],
                ex["y"],
                event_id
            )
            plot_csp_separation(ex["X_avg"], ex["y_avg"])

            ex["X"] = ex["epochs"].get_data()

            ex["clf"] = make_clf()

            ex["clf"].fit(ex["X_avg"], ex["y_avg"])
            dump_model(ex["clf"], ex["name"], amount_of_subjects)

            ex["crossval_scores"] = cross_val_score(
                ex["clf"], ex["X_avg"], ex["y_avg"], cv=cv,
                error_score='raise')
        score_and_print_results()
    else:
        args = parse_arguments()
        raw, epochs = read_subject(args.subject, args.run)

        plot_raw(raw)
        plot_psd(raw)
        raw_filtered = filter_raw(raw)
        plot_psd(raw_filtered)

        epochs = balance_classes(epochs)

        plot_evoked(epochs)
        plot_epochs_image(epochs)
        plot_compare_evokeds(epochs)

        X_avg = epochs.get_data()
        y = epochs.events[:, -1]

        X_avg, y_avg = average_over_epochs(epochs, y, epochs.event_id)
        plot_csp_separation(X_avg, y_avg)

        clf = make_clf()
        clf.fit(X_avg, y_avg)
        dump_model(clf, "S" + str(args.subject) + "R" + str(args.run), 1)

        crossval_scores = cross_val_score(
            clf, X_avg, y_avg, cv=cv, error_score='raise')
        print(crossval_scores)
        print("Crossval score: ", np.mean(crossval_scores))


if __name__ == "__main__":
    main()
