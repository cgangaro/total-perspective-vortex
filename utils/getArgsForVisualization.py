import argparse

from utils.ExternalFilesProcess import loadDatasetConfigFromJson
from utils.ExternalFilesProcess import loadPreProcessConfigFromJson


def getArgsForVisualization():

    parser = argparse.ArgumentParser(
        description='EEG Preprocessing Visualization'
    )

    parser.add_argument(
        '--subject',
        type=int,
        required=True,
        help="Subject ID to use for testing"
    )
    parser.add_argument(
        '--task',
        type=int,
        required=True,
        help="Task ID to use for testing"
    )
    parser.add_argument('--preprocess_config', type=str, required=False,
                        help='Path to the preprocess config file')
    parser.add_argument('--dataset_config', type=str, required=False,
                        help='Path to the dataset config file')
    parser.add_argument('--low_filter', type=float, required=False,
                        help='Low filter')
    parser.add_argument('--high_filter', type=float, required=False,
                        help='High filter')

    args = parser.parse_args()

    subject = args.subject
    task = args.task
    lowFilter = args.low_filter
    if lowFilter is None:
        lowFilter = 8.0
    highFilter = args.high_filter
    if highFilter is None:
        highFilter = 36.0

    datasetConfigFilePath = args.dataset_config
    if datasetConfigFilePath is None:
        datasetConfigFilePath = "config/dataset_config.json"

    preprocessConfigFilePath = args.preprocess_config
    if preprocessConfigFilePath is None:
        preprocessConfigFilePath = "config/preprocess_config.json"

    try:
        datasetConfig = loadDatasetConfigFromJson(datasetConfigFilePath)
    except Exception as e:
        raise Exception("Error loading dataset config failed: ", e)
    try:
        preProcessConfig = loadPreProcessConfigFromJson(
            preprocessConfigFilePath
        )
    except Exception as e:
        raise Exception("loading preprocess config failed: ", e)

    if subject is None or subject not in datasetConfig.subjects:
        raise Exception(f"Subject {subject} not in dataset")
    if task is None or task not in datasetConfig.tasks:
        raise Exception(f"Task {task} not in dataset")

    preProcessConfig.lowFilter = lowFilter
    preProcessConfig.highFilter = highFilter

    return subject, task, preProcessConfig
