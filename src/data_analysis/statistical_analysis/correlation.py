import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.helper_functions.general_helpers import load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
participants = np.arange(1, 23)

main_dataframe = load_pickle("main_dataframe.pickle")
main_dataframe_long = load_pickle("main_dataframe_long.pickle")
main_dataframe_long_2 = load_pickle("main_dataframe_long_2.pickle")
column_names = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_idle",
                "nasa_tlx", "performance"]  #, "hand_jerk"]
column_names_2 = ["overlap_grab_list", "ratio_frequency_list_items", "ratio_time_list_items"]


def correlation_matrix_means():
    means = []
    mean_dict = {}
    column_name_idx = 0
    for index, column in enumerate(main_dataframe.columns):
        means.append(main_dataframe[column].mean())
        if (index + 1) % 7 == 0:
            mean_dict[column_names[column_name_idx]] = means
            means = []
            column_name_idx += 1

    correlation_df = pd.DataFrame(mean_dict)

    matrix = correlation_df.corr().round(2)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag", mask=mask)
    plt.xticks(rotation=30)

    plt.show()


def correlation_matrix():
    measures_dataframe = main_dataframe_long[column_names]

    # sns.pairplot(measures_dataframe, hue="condition", markers=["o", "s", "D", "^", "v", "p", "*"])
    # plt.suptitle("Pair Plot of Measures by Condition")
    # plt.show()

    matrix = measures_dataframe.corr().round(2)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag", mask=mask)
    plt.xticks(rotation=30)
    plt.show()


if __name__ == "__main__":
    # correlation_matrix_means()
    correlation_matrix()

