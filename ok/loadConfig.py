import json
import os
from typing import List
from dataclassModels import Experiment, DatasetConfig, PreProcessConfiguration


def loadExperimentsFromJson(json_file: str) -> List[Experiment]:
    if not os.path.exists(json_file):
        raise Exception(f"File {json_file} does not exist")
    with open(json_file, 'r') as file:
        data = json.load(file)
        experiments = [Experiment(**item) for item in data]
    return experiments

def loadDatasetConfigFromJson(json_file: str) -> DatasetConfig:
    if not os.path.exists(json_file):
        raise Exception(f"File {json_file} does not exist")
    with open(json_file, 'r') as file:
        data = json.load(file)
        dataset_config = DatasetConfig(**data)
    return dataset_config

def loadPreProcessConfigFromJson(json_file: str) -> PreProcessConfiguration:
    if not os.path.exists(json_file):
        raise Exception(f"File {json_file} does not exist")
    with open(json_file, 'r') as file:
        data = json.load(file)
        config = PreProcessConfiguration.from_dict(data)
    return config