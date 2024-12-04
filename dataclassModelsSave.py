from dataclasses import asdict, dataclass
from typing import List


@dataclass
class DatasetConfig:
    tasks: List[int]
    subjects: List[int]


@dataclass
class Experiment:
    id: int
    name: str
    runs: List[int]


@dataclass
class PreProcessConfiguration:
    dataLocation: str = "/home/cgangaro/sgoinfre/mne_data"
    makeMontage: bool = True
    montageShape: str = "standard_1020"
    resample: bool = True
    resampleFreq: float = 90.0
    lowFilter: float = 8.0
    highFilter: float = 36.0
    ica: bool = True
    icaComponents: int = 20
    eog: bool = True
    epochsTmin: float = -1.0
    epochsTmax: float = 3.0

    @staticmethod
    def from_dict(config_dict):
        return PreProcessConfiguration(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class PredictArgs:
    subjects: List[int]
    tasks: List[int]
    dataDir: str
    modelsDir: str
    datasetConfig: DatasetConfig
    experimentsConfig: List[Experiment]
    preprocessConfig: PreProcessConfiguration
    playBack: bool

# @dataclass
# class TrainArgs:
#     subjects: List[int]
#     tasks: List[int]
#     trainDataDir: str
#     testDataDir: str
#     modelsDir: str
#     datasetConfig: DatasetConfig
#     experimentsConfig: List[Experiment]
#     preprocessConfig: PreProcessConfiguration
#     saveData: bool
#     loadData: bool

@dataclass
class TrainArgs:
    modelsDir: str
    datasetConfig: DatasetConfig
    experimentsConfig: List[Experiment]
    preprocessConfig: PreProcessConfiguration


class getArgsException(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"[Error getting program arguments]: {self.args[0]}"