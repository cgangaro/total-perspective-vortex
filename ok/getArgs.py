import argparse
import os
from loadConfig import loadDatasetConfigFromJson, loadExperimentsFromJson, loadPreProcessConfigFromJson
from dataclassModels import Args


class getArgsException(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"[Error getting program arguments]: {self.args[0]}"


def getArgs():
    try:
        parser = argparse.ArgumentParser(description='EEG Preprocessing and Classification - Predict')
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
        parser.add_argument('--data', type=str, required=False, help='Path to the data directory')
        parser.add_argument('--models', type=str, required=False, help='Path to the models directory')
        parser.add_argument('--config', type=str, required=False, help='Path to the dataset config file')
        parser.add_argument('--experiments', type=str, required=False, help='Path to the experiments config file')
        parser.add_argument('--preprocess_config', type=str, required=True, help='Path to the preprocess config file')
        
        args = parser.parse_args()

        subjects = args.subjects
        tasks = args.tasks
        dataDir = args.data
        modelsDir = args.models

        dataConfigFilePath = args.config
        if dataConfigFilePath is None:
            dataConfigFilePath = "ok/dataset_config.json"

        experimentsConfigFilePath = args.experiments
        if experimentsConfigFilePath is None:
            experimentsConfigFilePath = "ok/experiments.json"

        preprocessConfigFilePath = args.preprocess_config
        if preprocessConfigFilePath is None:
            preprocessConfigFilePath = "ok/preprocess_config.json"
    
        try:
            print("Loading dataset config from ", dataConfigFilePath)
            datasetConfig = loadDatasetConfigFromJson(dataConfigFilePath)
        except Exception as e:
            raise Exception("loading dataset config failed: ", e)
        
        if subjects is None and dataDir is None:
            raise Exception("Please provide at least one of the following arguments: subjects, data")
        
        if subjects is not None and dataDir is not None:
            subjects = None

        try:
            experimentsConfig = loadExperimentsFromJson(experimentsConfigFilePath)
        except Exception as e:
            raise Exception("loading experiments config failed: ", e)
        
        if tasks is None:
            tasks = datasetConfig.tasks
            experiments = experimentsConfig
        else:
            experiments = []
            for task in tasks:
                for exp in experimentsConfig:
                    if task in exp.tasks and exp not in experiments:
                        experiments.append(exp)
        
        if experiments is None or len(experiments) == 0:
            raise Exception("No experiments found")
        
        if dataDir is not None:
            if not os.path.exists(dataDir):
                raise Exception("Data directory not found")
            for exp in experiments:
                if not os.path.exists(os.path.join(dataDir, f"experiment_{exp.id}.pkl")):
                    raise Exception(f"Data directory for experiment {exp.id} (experiment_{exp.id}.pkl) not found")
        
        try:
            preProcessConfig = loadPreProcessConfigFromJson(preprocessConfigFilePath)
        except Exception as e:
            raise Exception("loading preprocess config failed: ", e)
        
        return Args(
            subjects=subjects,
            tasks=tasks,
            dataDir=dataDir,
            modelsDir=modelsDir,
            datasetConfig=datasetConfig,
            experiments=experiments,
            preprocessConfig=preProcessConfig
        )

    except Exception as e:
        raise getArgsException(e)












