import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

from preprocessing.main.edit_main_dataframe import edit_main_dataframe_1, transform_long_column_to_separate_columns
from src.preprocessing.helper_functions.general_helpers import load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
participants = np.arange(1, 23)
measure_names = ["AOI Cart", "AOI List", "AOI Main Shelf", "AOI Other Object", "Heart Rate",
                     "Heart Rate Variability",
                     "Head Acc.", "Mean Grab Time", "RMSE Hand Trajectory", "NASA-TLX", "Performance", "Hand Jerk",
                     "Head Idle"]
# measure_names = ["overlap_grab_list", "ratio_frequency_list_items", "ratio_time_list_items"]
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]
# units = ["$seconds$", "$seconds$", "$seconds$", "$seconds$", "$bpm$", "$bpm$", "$m/s^2$", "$seconds$", "$meters$",
#          "$-$", "$seconds/item$", "$m/s^3$", "$seconds$"]
units = ["$%$", "$-$", "$seconds/item$"]



def is_normal_shapiro(data):
    statistic, p_value = stats.shapiro(data)
    # print(f"statistic: {statistic}, p-value: {p_value}")
    return p_value >= 0.05


def is_normal_kstest(data):
    statistic, p_value = stats.kstest(data, "norm")
    # print(f"statistic: {statistic}, p-value: {p_value}")
    return p_value >= 0.05


def plot_distribution(data, column, plot_title, x_label):
    plt.hist(data, edgecolor="black", bins=10)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.savefig(f"{DATA_DIRECTORY}/images/histograms/{column}.png")
    # plt.show()


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


def calculate_cohens_d(long_dataframe, measure_column, baseline_condition=1, comparison_conditions=np.arange(2, 8)):
    baseline_data = long_dataframe[long_dataframe['condition'] == baseline_condition][measure_column]
    cohens_ds = []
    for condition in comparison_conditions:
        comparison_data = long_dataframe[long_dataframe['condition'] == condition][measure_column]
        mean_baseline = baseline_data.mean()
        mean_comparison = np.mean(comparison_data)
        pooled_std_dev = (baseline_data.std() ** 2 + comparison_data.std() ** 2) / 2
        cohens_d = (mean_comparison - mean_baseline) / np.sqrt(pooled_std_dev)
        cohens_ds.append(cohens_d)
    return cohens_ds


def all_values_to_latex(dataframe):
    measures, column_names_all, means, mins, maxes, stds, skewnesses, kurtosises, units_all = (
        [], [], [], [], [], [], [], [], []
    )

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
    print(latex_dataframe.to_latex(float_format="%.5g"))

    return



def histogram_all(dataframe):
    measure_counter = 0
    for i, column in enumerate(dataframe.columns):
        if i % 7 == 0 and i != 0:
            measure_counter += 1
        dataset = dataframe[column].values
        title = f"{measure_names[measure_counter]}, {condition_names[i % 7]}"
        unit = units[measure_counter]
        plot_distribution(dataset, column, title, unit)
        plt.cla()
        plt.clf()


if __name__ == "__main__":
    """
    measures analysis 1 = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
        "hand_grab_time", "hand_rmse", "nasa_tlx", "performance", "hand_jerk", "head_idle"]
        
    measures analysis 2 = ["overlap_grab_list", "ratio_frequency_list_items", "ratio_time_list_items"]
    """
    main_dataframe = load_pickle("main_dataframe_2.pickle")
    main_dataframe_columns = main_dataframe.columns

    measures_analysis_1 = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
         "hand_grab_time", "hand_rmse", "nasa_tlx", "performance", "hand_jerk", "head_idle"]
    measures_analysis_2 = ["overlap_grab_list", "ratio_frequency_list_items", "ratio_time_list_items"]

    long_df = load_pickle("main_dataframe_long.pickle")
    performance_df = transform_long_column_to_separate_columns(long_df, "performance")
    all_values_to_latex(performance_df)
    # plot_distribution(main_dataframe["performance_3"].values)

