from utils.preProcess import readDataset
from utils.getArgsForVisualization import getArgsForVisualization


def main():

    try:
        print("Preprocess data visualization")

        subject, task, preProcessConfig = getArgsForVisualization()

        print(f"Subject: {subject}, Task: {task}")

        mapping = {
            0: "Rest",
            1: "Move 1",
            2: "Move 2"
        }

        _ = readDataset(
            subjects=[subject],
            runs=[task],
            mapping=mapping,
            config=preProcessConfig,
            display=True
        )

        return 0

    except Exception as e:
        print("Error in preprocessVisualization program: ", e)
        return 1


if __name__ == "__main__":
    main()
