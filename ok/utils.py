import json
import os
from typing import List
import joblib
from preprocess import PreProcessConfiguration
from dataclasses import Experiment, DatasetConfig


def saveModels(models, save_directory="models"):
    os.makedirs(save_directory, exist_ok=True) 
    for exp_id, model in models.items():
        filename = os.path.join(save_directory, f"model_experiment_{exp_id}.joblib")
        joblib.dump(model, filename)
        print(f"Modèle de l'expérience {exp_id} sauvegardé dans {filename}.")


