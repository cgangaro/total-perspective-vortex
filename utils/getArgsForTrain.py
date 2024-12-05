import argparse
from utils.ExternalFilesProcess import loadDatasetConfigFromJson, loadExperimentsFromJson, loadPreProcessConfigFromJson
from utils.dataclassModels import TrainArgs, getArgsException


def getArgsForTrain():
    try:
        parser = argparse.ArgumentParser(description='Train EEG Classification Models')
        parser.add_argument('--dataset_config', type=str, required=False, help='Path to the dataset config file')
        parser.add_argument('--preprocess_config', type=str, required=False, help='Path to the preprocess config file')
        parser.add_argument('--experiments_config', type=str, required=False, help='Path to the experiments config file')
        parser.add_argument('--save_models_dir', type=str, required=False, help='Directory to save models')
        parser.add_argument('--split_by_subjects', action='store_true', help='Split the dataset by subjects')
        
        args = parser.parse_args()

        datasetConfigFilePath = args.dataset_config
        if datasetConfigFilePath is None:
            datasetConfigFilePath = "config/dataset_config.json"

        experimentsConfigFilePath = args.experiments_config
        if experimentsConfigFilePath is None:
            experimentsConfigFilePath = "config/experiments_config.json"

        preprocessConfigFilePath = args.preprocess_config
        if preprocessConfigFilePath is None:
            preprocessConfigFilePath = "config/preprocess_config.json"

        try:
            datasetConfig = loadDatasetConfigFromJson(datasetConfigFilePath)
        except Exception as e:
            raise Exception("Error loading dataset config failed: ", e)
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

        saveModelsDir = args.save_models_dir

        if saveModelsDir is None or saveModelsDir == "":
            saveModelsDir = "/home/cgangaro/goinfre/models"
        
        return TrainArgs(
            modelsDir=saveModelsDir,
            datasetConfig=datasetConfig,
            experimentsConfig=experimentsConfig,
            preprocessConfig=preProcessConfig,
            splitBySubjects=args.split_by_subjects
        )

    except Exception as e:
        raise getArgsException(e)













