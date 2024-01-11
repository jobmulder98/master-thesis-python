import numpy as np
from dotenv import load_dotenv
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.preprocessing.helper_functions.general_helpers import load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
participants = np.arange(1, 23)
# Measure names = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
#     "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"].


def is_normal_shapiro(data):
    statistic, p_value = stats.shapiro(data)
    # print(f"statistic: {statistic}, p-value: {p_value}")
    return p_value >= 0.05


def is_normal_kstest(data):
    statistic, p_value = stats.kstest(data, "norm")
    # print(f"statistic: {statistic}, p-value: {p_value}")
    return p_value >= 0.05


def plot_distribution(data):
    plt.hist(data, edgecolor="black", bins=10)
    plt.show()


def plot_scatter_measures(main_dataframe, measure):
    data = {}
    for i in range(1, 8):
        current_measure = f"{measure}_{i}"
        x = main_dataframe[current_measure]
        y = np.ones(len(x)) * i
        data[current_measure] = (x, y)

    for key, value in data.items():
        plt.scatter(value[0], value[1])
    plt.show()


if __name__ == "__main__":
    """
    Measure names = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
        "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"].
    """
    main_dataframe = load_pickle("main_dataframe.pickle")
    # plot_scatter_measures(main_dataframe, "aoi_other_object")
    plot_distribution(main_dataframe["performance_4"].values)
    # for column in main_dataframe.columns:
    #     data = main_dataframe[column].values
        # print(f"The data of {column} is normally distributed: {is_normal_kstest(data)}")

