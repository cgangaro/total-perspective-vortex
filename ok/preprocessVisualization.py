import argparse
from loadConfig import loadDatasetConfigFromJson, loadPreProcessConfigFromJson
from preprocess import preprocessOneSubjectOneExperiment


def main():

    try:
        print("Preprocess data visualization")
        parser = argparse.ArgumentParser(description='EEG Preprocessing Visualization')
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
        parser.add_argument('--dataset_config', type=str, required=False, help='Path to the dataset config file')
        parser.add_argument('--preprocess_config', type=str, required=True, help='Path to the preprocess config file')
        
        args = parser.parse_args()
        
        subject = args.subject
        task = args.task
        datasetConfigFilePath = args.dataset_config
        if dataConfigFilePath is None:
            dataConfigFilePath = "ok/dataset_config.json"
        try:
            datasetConfig = loadDatasetConfigFromJson(dataConfigFilePath)
        except Exception as e:
            raise Exception("Error loading dataset config failed: ", e)
        preprocessConfigFilePath = args.preprocess_config
        try:
            preProcessConfig = loadPreProcessConfigFromJson(preprocessConfigFilePath)
        except Exception as e:
            raise Exception("loading preprocess config failed: ", e)
        
        if subject not in datasetConfig.subjects:
            raise Exception(f"Subject {subject} not in dataset")
        if task not in datasetConfig.tasks:
            raise Exception(f"Task {task} not in dataset")
        
        res = preprocessOneSubjectOneExperiment(
            subject=subject,
            task=[task],
            config=datasetConfig,
            display=True,
        )

    except Exception as e:
        print("Error in preprocessVisualization program: ", e)
        return 1
        
if __name__ == "__main__":
    main()
