import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)

fig, ax = plt.subplots()
plot_dictionary = {}

feature = "nasa-tlx unweighted"

for condition in conditions:
    with open(f"{DATA_DIRECTORY}\pickles\c{condition}.pickle", "rb") as handle:
        p = pickle.load(handle)
    filtered_data = [x for x in p[feature] if x is not None]
    plot_dictionary[condition] = filtered_data

print(plot_dictionary)
ax.set_title(feature)
ax.set_xlabel("condition")
ax.set_ylabel("total time (s)")
ax.boxplot(plot_dictionary.values())
ax.set_xticklabels(plot_dictionary.keys())
plt.show()
