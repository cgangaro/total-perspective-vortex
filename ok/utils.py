import json
import glob
import os
from preprocess import PreProcessConfiguration

def load_configurations(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
        config = PreProcessConfiguration.from_dict(config_dict)
        configurations.append(config)
    return configurations
