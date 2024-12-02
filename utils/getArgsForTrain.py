import argparse
from utils.ExternalFilesProcess import loadDatasetConfigFromJson, loadExperimentsFromJson, loadPreProcessConfigFromJson
from dataclassModels import TrainArgs, getArgsException


def getArgsForTrain():
    try:
        parser = argparse.ArgumentParser(description='Train EEG Classification Models')
        parser.add_argument('--dataset_config', type=str, required=False, help='Path to the dataset config file')
        parser.add_argument('--preprocess_config', type=str, required=False, help='Path to the preprocess config file')
        parser.add_argument('--experiments', type=str, required=False, help='Path to the experiments config file')
        parser.add_argument('--load_data', action='store_true', help='Charger les données prétraitées')
        parser.add_argument('--save_data', action='store_true', help='Enregistrer les données prétraitées')
        parser.add_argument('--train_data_dir', type=str, help='Data directory to save/load preprocessed train data')
        parser.add_argument('--test_data_dir', type=str, help='Data directory to save/load preprocessed test data')
        parser.add_argument('--save_models_dir', type=str, help='Directory to save models')
        
        args = parser.parse_args()

        datasetConfigFilePath = args.dataset_config
        if datasetConfigFilePath is None:
            datasetConfigFilePath = "ok/dataset_config.json"

        experimentsConfigFilePath = args.experiments
        if experimentsConfigFilePath is None:
            experimentsConfigFilePath = "ok/experiments.json"

        preprocessConfigFilePath = args.preprocess_config
        if preprocessConfigFilePath is None:
            preprocessConfigFilePath = "ok/preprocess_config.json"

        try:
            datasetConfig = loadDatasetConfigFromJson(datasetConfigFilePath)
        except Exception as e:
            raise Exception("Error loading dataset config failed: ", e)
        preprocessConfigFilePath = args.preprocess_config
        try:
            preProcessConfig = loadPreProcessConfigFromJson(preprocessConfigFilePath)
        except Exception as e:
            raise Exception("loading preprocess config failed: ", e)
        try:
            experimentsConfig = loadExperimentsFromJson(experimentsConfigFilePath)
        except Exception as e:
            raise Exception("loading experiments config failed: ", e)
        
        if datasetConfig.subjects is None or len(datasetConfig.subjects) <= 0:
            raise Exception("No subjects in dataset config")
        if datasetConfig.tasks is None or len(datasetConfig.tasks) <= 0:
            raise Exception("No tasks in dataset config")
        if experimentsConfig is None or len(experimentsConfig) <= 0:
            raise Exception("No experiments in experiments config")

        trainDataDir = args.train_data_dir
        testDataDir = args.test_data_dir
        saveModelsDir = args.save_models_dir

        if trainDataDir is None or trainDataDir == "":
            trainDataDir = "/home/cgangaro/goinfre/trainDataSave"
        if testDataDir is None or testDataDir == "":
            testDataDir = "/home/cgangaro/goinfre/trainDataSave"
        if saveModelsDir is None or saveModelsDir == "":
            saveModelsDir = "/home/cgangaro/goinfre/models"
        
        return TrainArgs(
            subjects=datasetConfig.subjects,
            tasks=datasetConfig.tasks,
            trainDataDir=trainDataDir,
            testDataDir=testDataDir,
            modelsDir=saveModelsDir,
            datasetConfig=datasetConfig,
            experiments=experimentsConfig,
            preprocessConfig=preProcessConfig
        )

    except Exception as e:
        raise getArgsException(e)













