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


main_dataframe = load_pickle("main_dataframe.pickle")
for column in main_dataframe.columns:
    data = main_dataframe[column].values
    print(f"The data of {column} is normally distributed: {is_normal_kstest(data)}")

