import os
import joblib
from sklearn.metrics import accuracy_score
from utils.preProcess import preprocessOneExperiment
from utils.getArgsForPredict import getArgsForPredict, getArgsException


def main():

    try:
        print("Predict program")
        try:
            args = getArgsForPredict()
        except getArgsException as e:
            print(e)
            return 1

        subjects = args.subjects
        experiments = args.experimentsConfig

        print(f"Subjects: {subjects}\n")

        models = {}
        for exp in experiments:
            print(f"Loading model for experiment {exp.id}")
            modelPath = f"{args.modelsDir}/model_experiment_{exp.id}.joblib"
            if not os.path.exists(modelPath):
                raise Exception(f"Model file {modelPath} not found")
            model = joblib.load(modelPath)
            models[exp.id] = model
        print()
        totalAccuracy = 0
        for i, exp in enumerate(experiments):

            print(f"Experiment: {exp.id} - {exp.name}")
            print(f"Runs: {exp.runs}")
            print(f"Mapping: {exp.mapping}")

            x, y = preprocessOneExperiment(
                args.subjects,
                exp.runs,
                exp.mapping,
                args.preprocessConfig,
                balance=False,
                average=False
            )
            pipeline = models[exp.id]

            if args.playBack:
                print(f"x[i] shape: {x[0]}")
                for i in range(len(x)):
                    p = pipeline.predict(x[i])
                    print(f"Epoch {i + 1}/{len(x)} - Label: {y[i]} -"
                          f" Prediction: {p[0]}")

            predictions = pipeline.predict(x)
            accuracy = accuracy_score(y, predictions)
            totalAccuracy += accuracy
            print(f"Experiment {exp.id} - Test Accuracy: {accuracy:.2f} on "
                  f"{len(x)} epochs, {len(y)} labels\n")
        totalAccuracy /= len(experiments)
        print(f"Total Accuracy: {totalAccuracy:.4f}")

    except Exception as e:
        print("Error in predict program: ", e)
        return 1


if __name__ == "__main__":
    main()
