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
from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
import numpy as np
import mne
from newPreprocess import newPreprocess, Configuration, PreProcessConfiguration
matplotlib.use("webagg")


def main():

    config = PreProcessConfiguration(
        withTargetSfreq=True,
        makeMontage=True,
        montageShape='standard_1005',
        lowFilter=8.0,
        highFilter=32.0
    )

    subjects1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    subjects2 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    subjects3 = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    subjects = subjects1 + subjects2 + subjects3


    dataPreprocessed = newPreprocess(subjects, config)

    for data in dataPreprocessed:
        print(data['epochs'])
        print(data['labels'])
        epochs = data['epochs']
        labels = data['labels']
        score_pipeline = -1
        models_pipeline = None

        # Define a monte-carlo cross-validation generator (reduce variance):
        scores = []
        epochs_data = epochs.get_data()
        cv = ShuffleSplit(10, test_size=0.4, random_state=42)

        # # Assemble a classifier
        csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
        rfc = RandomForestClassifier(n_estimators=150, random_state=42)
        print(f"Avant pipeline : epochs_data type: {epochs_data.dtype}, shape: {epochs_data.shape}")
        print(f"Labels type: {labels.dtype}, shape: {labels.shape}")
        # Use scikit-learn Pipeline with cross_val_score function
        clf = make_pipeline(csp, rfc)
        scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

        # Printing the results
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        if scores.mean() > score_pipeline:
            score_pipeline = scores.mean()
            models_pipeline = clf

        # plot CSP patterns estimated on full data for visualization
        csp.fit_transform(epochs_data, labels)
        # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
        # plt.show()

    return

if __name__ == "__main__":
    main()