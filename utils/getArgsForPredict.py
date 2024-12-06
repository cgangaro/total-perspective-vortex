import argparse
from utils.ExternalFilesProcess import (
    loadDatasetConfigFromJson,
    loadExperimentsFromJson,
    loadPreProcessConfigFromJson
)
from utils.dataclassModels import PredictArgs, Experiment, getArgsException


def getArgsForPredict():
    try:
        parser = argparse.ArgumentParser(
            description='EEG Preprocessing and Classification - Predict'
        )
        parser.add_argument(
            '--subjects',
            type=int,
            nargs='+',
            required=False,
            help="Subjects ID to use for testing"
        )
        parser.add_argument(
            '--tasks',
            type=int,
            nargs='+',
            required=False,
            help="Tasks ID to use for testing"
        )
        parser.add_argument('--models', type=str, required=False,
                            help='Path to the models directory')
        parser.add_argument('--experiments_config', type=str, required=False,
                            help='Path to the experiments config file')
        parser.add_argument('--preprocess_config', type=str, required=False,
                            help='Path to the preprocess config file')
        parser.add_argument('--dataset_config', type=str, required=False,
                            help='Path to the dataset config file')
        parser.add_argument('--play_back', action='store_true',
                            help='Play back the EEG data')

        args = parser.parse_args()

        subjects = args.subjects
        tasks = args.tasks
        modelsDir = args.models

        dataConfigFilePath = args.dataset_config
        if dataConfigFilePath is None:
            dataConfigFilePath = "config/dataset_config.json"

        experimentsConfigFilePath = args.experiments_config
        if experimentsConfigFilePath is None:
            experimentsConfigFilePath = "config/experiments_config.json"

        preprocessConfigFilePath = args.preprocess_config
        if preprocessConfigFilePath is None:
            preprocessConfigFilePath = "config/preprocess_config.json"
        
        if modelsDir is None or modelsDir == "":
            modelsDir = "/home/cgangaro/goinfre/models"
    
        try:
            datasetConfig = loadDatasetConfigFromJson(dataConfigFilePath)
        except Exception as e:
            raise Exception("loading dataset config failed: ", e)
        
        if subjects is None:
            subjects = datasetConfig.subjects

        try:
            experimentsConfig = loadExperimentsFromJson(
                experimentsConfigFilePath
            )
        except Exception as e:
            raise Exception("loading experiments config failed: ", e)
        
        if tasks is None:
            tasks = datasetConfig.tasks
        experiments = getExperimentsWithThisTasks(experimentsConfig, tasks)
        
        if experiments is None or len(experiments) == 0:
            raise Exception("No experiments found")
        
        try:
            preProcessConfig = loadPreProcessConfigFromJson(
                preprocessConfigFilePath
            )
        except Exception as e:
            raise Exception("loading preprocess config failed: ", e)
        
        return PredictArgs(
            subjects=subjects,
            tasks=tasks,
            modelsDir=modelsDir,
            datasetConfig=datasetConfig,
            experimentsConfig=experiments,
            preprocessConfig=preProcessConfig,
            playBack=args.play_back
        )

    except Exception as e:
        raise getArgsException(e)


def getExperimentsWithThisTasks(experiments, tasks):
    tasks = set(tasks)
    newExperiments = []
    for task in tasks:
        experimentsTmp = experiments.copy()
        expFound = next((exp for exp in experimentsTmp if task in exp.runs),
                        None)
        while expFound is not None:
            experimentsTmp.remove(expFound)
            newExp = next(
                (newExp for newExp in newExperiments
                 if expFound.id == newExp.id),
                None
            )
            if newExp is None:
                newExperiments.append(
                    Experiment(
                        id=expFound.id,
                        name=expFound.name,
                        runs=[task],
                        mapping=expFound.mapping
                    )
                )
            else:
                if task not in newExp.runs:
                    newExp.runs.append(task)
            expFound = next((exp for exp in experimentsTmp
                             if task in exp.runs),
                            None)
    newExperiments.sort(key=lambda x: x.id)
    return newExperiments
