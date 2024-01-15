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
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]

fig, ax = plt.subplots()
plot_dictionary = {}

# feature = "nasa-tlx unweighted"
feature = "nasa-tlx weighted"
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
# feature = "mean head acceleration"
# feature = "rmse trajectory item to cart"
# feature = "mean grab time"

for condition in conditions:
    with open(f"{DATA_DIRECTORY}\pickles\c{condition}.pickle", "rb") as handle:
        p = pickle.load(handle)
    filtered_data = [x for x in p[feature] if x is not None]
    condition_name = condition_names[condition-1]
    plot_dictionary[condition_name] = filtered_data

if feature == "nasa-tlx weighted":
    ax.set_title("Weighted NASA-TLX")
else:
    ax.set_title(f"{feature.title()} for all conditions")
ax.set_xlabel("Condition")
ax.set_ylabel("Weighted NASA-TLX Score")
ax.boxplot(plot_dictionary.values())
ax.set_xticklabels(condition_names)
fig.autofmt_xdate(rotation=30)
# mean_std = []
# for key, value in plot_dictionary.items():
#     mean_std.append(f"{np.round(np.mean(value), 3)} ({np.round(np.std(value), 3)})")
# data = pd.DataFrame({"means": mean_std})
# data.to_excel(f"{DATA_DIRECTORY}/other/dummy.xlsx")
plt.show()
