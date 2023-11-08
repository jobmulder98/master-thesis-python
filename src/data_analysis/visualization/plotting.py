import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from src.preprocessing.preprocess import initialize_synchronized_times
from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda
from src.preprocessing.ecg_eda.ecg.features import ecg_features

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)

fig, ax = plt.subplots()
plot_dictionary = {}

feature = "total time other object"

for condition in conditions:
    with open(f"{DATA_DIRECTORY}\pickles\c{condition}.pickle", "rb") as handle:
        p = pickle.load(handle)
    plot_dictionary[condition] = p[feature]

ax.set_title(feature)
ax.set_xlabel("condition")
ax.set_ylabel("seconds")
ax.boxplot(plot_dictionary.values())
ax.set_xticklabels(plot_dictionary.keys())
plt.show()
