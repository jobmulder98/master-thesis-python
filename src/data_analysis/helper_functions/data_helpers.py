import pickle

import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def obtain_feature_data(feature, conditions):
    feature_dictionary = {}
    for condition in conditions:
        with open(f"{DATA_DIRECTORY}\pickles\c{condition}.pickle", "rb") as handle:
            p = pickle.load(handle)
        filtered_data = [x for x in p[feature] if x is not None]
        feature_dictionary[condition] = filtered_data
    return feature_dictionary
