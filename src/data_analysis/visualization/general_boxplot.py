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
condition_names = ["no stimuli", "visual low", "visual high", "auditory low", "auditory high", "mental low", "mental high"]

fig, ax = plt.subplots()
plot_dictionary = {}

# feature = "nasa-tlx unweighted"
# feature = "nasa-tlx weighted"
# feature = "all fixations"
# feature = "mean fixation time"
# feature = "total time other object"
# feature = "fixations other object"
# feature = "seconds/item first 16"
# feature = "seconds/item window"
# feature = "std dev. seconds/item first 16"
# feature = "total time other object"
# feature = "total time main shelf"
# feature = "total time list"
# feature = "total time cart"
feature = "mean head acceleration"

for condition in conditions:
    with open(f"{DATA_DIRECTORY}\pickles\c{condition}.pickle", "rb") as handle:
        p = pickle.load(handle)
    filtered_data = [x for x in p[feature] if x is not None]
    condition_name = condition_names[condition-1]
    plot_dictionary[condition_name] = filtered_data

ax.set_title(feature.title())
ax.set_xlabel("Condition")
fig.autofmt_xdate(rotation=45)
ax.set_ylabel("Mean Acceleration (m/s^2)")
ax.boxplot(plot_dictionary.values())
ax.set_xticklabels(plot_dictionary.keys())
plt.show()
