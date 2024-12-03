import os
import joblib
from sklearn.metrics import accuracy_score
from utils.preprocess import preprocess
from utils.getArgsForPredict import getArgsForPredict, getArgsException


def main():

    try:
        print("Predict program")
        try:
            args = getArgsForPredict()
        except getArgsException as e:
            print(e)
            return 1
        
        print(f"Args: {args}")
        print(f"Experiments: {args.experimentsConfig}")
        print("\n\n-----Preprocess data-----\n")
        if args.dataDir is not None and args.dataDir != "":
            dataPreprocessed = preprocess(
                    saveDirectory=args.dataDir,
                    config=args.preprocessConfig,
                    experiments=args.experimentsConfig,
                    loadData=True
                )
        else:
            dataPreprocessed = preprocess("", args.preprocessConfig, args.subjects, args.experimentsConfig)
        

        print("\n\n-----Load models-----\n")
        models = {}
        for exp in args.experimentsConfig:
            modelPath = f"{args.modelsDir}/model_experiment_{exp.id}.joblib"
            if not os.path.exists(modelPath):
                raise Exception(f"Model file {modelPath} not found")
            model = joblib.load(modelPath)
            models[exp.id] = model

        print("\n\n-----Predict-----\n")
        print(f"Test data: {len(dataPreprocessed)} experiments, models: {len(models)}")

        accuracyTotal = 0
        for dataTest in dataPreprocessed:
            expId = dataTest['experiment']
            epochs = dataTest['epochs']
            labels = dataTest['labels']
            subject_ids = dataTest['subject_ids']
            epochs_data = epochs.get_data()
            clf = models[expId]

            if args.playBack:
                print(f"Experiment {expId} - {epochs_data.shape} epochs, labels: {labels.shape}")
                for i in range(len(epochs)):
                    p = clf.predict(epochs_data[i].reshape(1, -1))
                    print(f"Epoch {i}/{len(epochs)} - Label: {labels[i]} - Prediction: {p[0]}")
            predictions = clf.predict(epochs_data)
            accuracy = accuracy_score(labels, predictions)
            print(f"Experiment {expId} - Test Accuracy: {accuracy:.2f} on {len(epochs)} epochs, {len(labels)} labels and ({len(set(subject_ids))}) subjects")
            accuracyTotal += accuracy
        accuracyTotal /= len(dataPreprocessed)
        print(f"Total Accuracy: {accuracyTotal:.4f}")

    except Exception as e:
        print("Error in predict program: ", e)
        return 1
        
if __name__ == "__main__":
    main()


