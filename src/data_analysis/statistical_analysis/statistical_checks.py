import numpy as np
import pandas as pd
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


def skewness(data):
    return stats.skew(data, axis=0, bias=True)


def kurtosis(data):
    return stats.kurtosis(data, axis=0, fisher=True, bias=True)


def all_values_to_latex(dataframe):
    measures, column_names_all, means, mins, maxes, stds, skewnesses, kurtosises, units_all = (
        [], [], [], [], [], [], [], [], []
    )

    measure_names = ["AOI Cart", "AOI List", "AOI Main Shelf", "AOI Other Object", "Heart Rate",
                     "Heart Rate Variability",
                     "Head Acc.", "Mean Grab Time", "RMSE Hand Trajectory", "NASA-TLX", "Performance"]
    condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                       "Mental High"]
    units = ["$seconds$", "$seconds$", "$seconds$", "$seconds$", "$bpm$", "$bpm$", "$m/s^2$", "$seconds$", "$meters$",
             "$-$", "$seconds/item$"]

    measure_counter = 0
    for i, column in enumerate(dataframe.columns):
        if i % 7 == 0 and i != 0:
            measure_counter += 1

        dataset = dataframe[column].values

        measures.append(measure_names[measure_counter])
        column_names_all.append(condition_names[i % 7])
        means.append(np.mean(dataset).round(3))
        mins.append(np.min(dataset).round(3))
        maxes.append(np.max(dataset).round(3))
        stds.append(np.std(dataset).round(3))
        skewnesses.append(skewness(dataset).round(3))
        kurtosises.append(kurtosis(dataset).round(3))
        units_all.append(units[measure_counter])

    latex_dictionary = {
        "Condition": column_names_all,
        "Mean": means,
        "SD": stds,
        "Min.": mins,
        "Max.": maxes,
        "Skewness": skewnesses,
        "Kurtosis": kurtosises,
        "Unit": units_all,
    }
    latex_dataframe = pd.DataFrame(latex_dictionary)
    latex_dataframe = latex_dataframe.set_index("Condition")
    print(latex_dataframe.to_latex(float_format="%.3f"))

    return


if __name__ == "__main__":
    """
    measures analysis 1 = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
        "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"]
        
    measures analysis 2 = ["overlap_grab_list", "ratio_frequency_list_items", "ratio_time_list_items"]
    """
    main_dataframe = load_pickle("main_dataframe.pickle")

    rmse_values = main_dataframe["hand_rmse_1"].values
    rmse_values[9] = np.mean(rmse_values)

    print(np.mean(rmse_values), np.std(rmse_values), np.min(rmse_values), np.max(rmse_values), skewness(rmse_values), kurtosis(rmse_values))

    # all_values_to_latex(main_dataframe)
    # plot_distribution(main_dataframe["performance_3"].values)


