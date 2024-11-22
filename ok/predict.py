import json
import random
import argparse
import os
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from mne.decoding import CSP
from CSP import CSP
from WaveletFeatureExtractor import WaveletFeatureExtractor
from preprocess import PreProcessConfiguration, preprocess, Experiment
from WaveletTransformer import WaveletTransformer
from getArgs import getArgs, getArgsException


def main():

    try:
        print("Predict program")
        try:
            args = getArgs()
        except getArgsException as e:
            print(e)
            return 1
        
        print(f"Args: {args}")
        print(f"Experiments: {args.experiments}")
        print("\n\n-----Preprocess data-----\n")
        if args.dataDir is not None and args.dataDir != "":
            dataPreprocessed = preprocess(
                    saveDirectory=args.dataDir,
                    config=args.preprocessConfig,
                    experiments=args.experiments,
                    loadData=True
                )
        else:
            dataPreprocessed = preprocess("", args.preprocessConfig, args.subjects, args.experiments)
        

        print("\n\n-----Load models-----\n")
        models = {}
        for exp in args.experiments:
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


