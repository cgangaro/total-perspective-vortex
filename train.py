import random
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.CSP import CSP
from utils.preprocess import preprocess
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
        
        dataTrainPreprocessed = preprocess(
            saveDirectory=args.trainDataDir,
            config=args.preProcessConfig,
            subjects=trainSubjects,
            experiments=args.experimentsConfig,
            saveData=args.save_data,
            loadData=args.load_data
        )
        dataTestPreprocessed = preprocess(
            saveDirectory=args.testDataDir,
            config=args.preProcessConfig,
            subjects=testSubjects,
            experiments=args.experimentsConfig,
            saveData=args.save_data,
            loadData=args.load_data
        )

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
        if saveModelsDir is None or saveModelsDir == "":
            saveModelsDir = "models"
        saveModels(models, saveModelsDir)


    except Exception as e:
        print("Error in train program: ", e)
        return 1


if __name__ == "__main__":
    main()
