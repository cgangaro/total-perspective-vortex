from joblib import load
import argparse
import numpy as np
import time

from config import MODELS_PATH

from read_dataset import read_subject
from read_dataset import check_subject_and_run


def load_model(name):
    print("Loading model", name, "...")
    model = load(MODELS_PATH + name + "_109_subjects.joblib")
    return model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict the classes of a given EEG signal"
    )
    parser.add_argument("model", type=str, help="Name of the model to use")
    parser.add_argument("subject", type=int, help="Subject number")
    parser.add_argument("run", type=int, help="Run number")
    args = parser.parse_args()
    check_subject_and_run(args.subject, args.run)
    return args


def predict(model, subject, run):
    print("Predicting subject", subject, "run", run)

    raw, epochs = read_subject(subject, run)
    actual = epochs.events[:, -1]
    print("Epoch:\t\t[Prediction]\t[Truth]\tEqual?\tTime to predict")
    for i, e in enumerate(epochs.get_data()):
        start = time.process_time()
        prediction = model.predict(np.array([e]))
        time_to_pred = time.process_time() - start + 1
        print("Epoch %2d :\t%d\t\t%d\t%s\t%.5f" %
              (i, prediction, actual[i], prediction == actual[i],
               time_to_pred))
        time.sleep(1)
    print("Accuracy: %.4f" % model.score(epochs.get_data(), actual))


def main():
    args = parse_arguments()
    model = load_model(args.model)
    print(model)
    predict(model, args.subject, args.run)


if __name__ == "__main__":
    main()
