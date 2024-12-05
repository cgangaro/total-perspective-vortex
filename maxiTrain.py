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
from utils.ExternalFilesProcess import saveModels
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

        if args.splitBySubjects:
            random.shuffle(subjects)
            sizeTestTab = int(len(subjects) * 0.3)
            testSubjects = subjects[:sizeTestTab]
            trainSubjects = subjects[sizeTestTab:]
            print(f"{len(trainSubjects)} train subjects, {len(testSubjects)} test subjects")
        else:
            trainSubjects = subjects

        print("\n\n----------TRAIN----------\n")
        expData = []
        for i, exp in enumerate(experiments):

            print(f"Experiment: {exp.id} - {exp.name}")
            print(f"Runs: {exp.runs}")
            print(f"Mapping: {exp.mapping}")

            if args.splitBySubjects:
                xTrainAvg, yTrainAvg = preprocessOneExperiment(trainSubjects, exp.runs, exp.mapping, args.preprocessConfig, balance=True, average=True)
                xTest, yTest = preprocessOneExperiment(testSubjects, exp.runs, exp.mapping, args.preprocessConfig, balance=False, average=False)
            else:
                xTrainAvg, yTrainAvg, xTest, yTest = preprocessOneExperiment(trainSubjects, exp.runs, exp.mapping, args.preprocessConfig, balance=True, average=True, splitData=True)
            
            csp = MyCSP(n_components=8)
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')

            pipeline = Pipeline([
                ("CSP", csp),
                ("LDA", lda)
            ])
            # pipeline = Pipeline([
            #     ("CSP", MyCSP(n_components=4)),
            #     ("LDA", LinearDiscriminantAnalysis(solver='svd', tol=0.0001))
            # ])

            pipeline.fit(xTrainAvg, yTrainAvg)

            cv = ShuffleSplit(10, test_size=0.2, random_state=42)

            score = cross_val_score(
                estimator=pipeline,
                X=xTrainAvg,
                y=yTrainAvg,
                cv=cv,
                error_score='raise'
            )
            print(f"Crossval scores: {score}\n")
            
            expData.append({
                "id": exp.id,
                "name": exp.name,
                "xTest": xTest,
                "yTest": yTest,
                "xTrain": xTrainAvg,
                "yTrain": yTrainAvg,
                "pipeline": pipeline,
                "score": score
            })

        print("\n\n----------TEST----------\n")
        if args.splitBySubjects:
            print(f"Test data: {len(testSubjects)} subjects. Subjects: {testSubjects}\n")
        test_scores = []
        train_scores = []
        crossval_scores = []
        print()

        for data in expData:
            id = data["id"]
            name = data["name"]
            X_test = data["xTest"]
            y_test = data["yTest"]
            train_score = data["pipeline"].score(data["xTrain"], data["yTrain"])
            test_score = data["pipeline"].score(X_test, y_test)

            print(f"Experiment {id} - {name}")
            print("Train: ", train_score)
            train_scores.append(train_score)

            print("Test: ", test_score)
            test_scores.append(test_score)

            print("Crossval: %f" % (np.mean(data["score"])))
            crossval_scores.append(np.mean(data["score"]))

            print()

        saveModels({exp["id"]: exp["pipeline"] for exp in expData}, args.modelsDir)
        print("Mean scores")
        print("Train: ", round(sum(train_scores) / len(train_scores), 2))
        print("Test: ", round(sum(test_scores) / len(test_scores), 2))
        print("Crossval: ", round(sum(crossval_scores) / len(crossval_scores), 2))

        return 0

    except Exception as e:
        print("Error in train program: ", e)
        return 1


if __name__ == "__main__":
    main()
