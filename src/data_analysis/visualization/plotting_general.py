import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import seaborn as sns

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]

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
# feature = "mean head acceleration"
# feature = "rmse trajectory item to cart"
# feature = "mean grab time"


def box_plot_general(feature):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for condition in conditions:
        with open(f"{DATA_DIRECTORY}\pickles\c{condition}.pickle", "rb") as handle:
            p = pickle.load(handle)
        if feature == "seconds/item window":
            filtered_data = [60 / x for x in p[feature] if x is not None]
        else:
            filtered_data = [x for x in p[feature] if x is not None]
        condition_name = condition_names[condition-1]
        plot_dictionary[condition_name] = filtered_data

    if feature == "nasa-tlx weighted":
        ax.set_title("Weighted NASA-TLX Score")
    elif feature == "seconds/item window":
        ax.set_title("Performance (Items per Minute)")
    else:
        ax.set_title(f"{feature.title()} for all conditions")

    data = pd.DataFrame(plot_dictionary)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Score (-)")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    plt.show()

