import pickle

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
        feature_dictionary[condition] = p[feature]
    return feature_dictionary
