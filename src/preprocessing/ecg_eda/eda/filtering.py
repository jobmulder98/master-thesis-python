import numpy as np
from dotenv import load_dotenv
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import neurokit2 as nk
from scipy.ndimage import median_filter

from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda
from src.preprocessing.helper_functions.general_helpers import (format_time,
                                                                interpolate_nan_values,
                                                                write_pickle,
                                                                load_pickle)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
EDA_SAMPLE_RATE = int(os.getenv("EDA_SAMPLE_RATE"))

participants = np.arange(1, 23)
conditions = np.arange(1, 8)
synchronized_times = load_pickle("synchronized_times.pickle")


def replace_values_above_threshold_to_nan(signal, threshold):
    filtered_signal = np.where(signal <= threshold, signal, np.nan)
    filtered_signal = np.where(filtered_signal >= -threshold, signal, np.nan)
    return filtered_signal


def add_margin_around_nan_values(signal, margin_in_milliseconds: int):
    filtered_signal = np.copy(signal)
    margin_in_indices = int(EDA_SAMPLE_RATE * (margin_in_milliseconds / 1000))
    nan_indices = np.where(np.isnan(signal))[0]
    for index in nan_indices:
        min_margin = max(0, index - margin_in_indices)
        max_margin = min(index + margin_in_indices + 1, len(filtered_signal))
        filtered_signal[min_margin:max_margin] = np.nan
    return filtered_signal


def one_dimensional_median_filter(signal, window_size_in_milliseconds):
    window_size = int(EDA_SAMPLE_RATE * (window_size_in_milliseconds / 1000))
    filtered_signal = median_filter(signal, size=window_size, mode='reflect')
    return filtered_signal


def decompose_eda_signal(signal):
    return nk.eda_phasic(nk.standardize(signal), sampling_rate=EDA_SAMPLE_RATE)


def filter_eda_data(eda_signal, peak_threshold, peak_margin, filter_window_size):
    signal_peaks_removed = replace_values_above_threshold_to_nan(eda_signal, peak_threshold)
    signal_peaks_removed_with_margin = add_margin_around_nan_values(signal_peaks_removed, peak_margin)
    interpolated_signal = np.array(interpolate_nan_values(signal_peaks_removed_with_margin.tolist()))
    decomposed_signal = decompose_eda_signal(interpolated_signal)
    tonic_signal = decomposed_signal["EDA_Tonic"].values
    phasic_signal = decomposed_signal["EDA_Phasic"].values
    # filtered_signal = one_dimensional_median_filter(interpolated_signal, filter_window_size)
    return interpolated_signal, tonic_signal, phasic_signal


def plot_conditions_one_participant(participant):
    synchronized_times = load_pickle("synchronized_times.pickle")
    for condition in np.arange(1, 8):
        start_index, end_index = synchronized_times[participant][condition]
        df = create_clean_dataframe_ecg_eda(participant).iloc[start_index:end_index]
        eda_signal = np.abs(df["Sensor-C:SC/GSR"].values)
        selected_values = eda_signal[::32]
        filtered_eda_signal, tonic_signal, phasic_signal = filter_eda_data(selected_values,
                                                                           1000,
                                                                           1000,
                                                                           200
                                                                           )
        plt.plot(filtered_eda_signal, label=f"condition {condition}")
    plt.legend()
    plt.show()


def compute_normalized_means_medians():
    """
    This function computes the mean and median for all conditions for each participant. After computing these
    values, the values are normalized, so they could be compared to other participants. Without normalizing
    within subject, the values differ too much and drawing conclusions is very difficult.

    Returns: box plot with normalized means and medians
    """
    eda_dictionary = load_pickle("eda_signal_filtered_normalized.pickle")
    synchronized_times = load_pickle("synchronized_times.pickle")
    normalized_means_medians = dict()
    for participant in participants:
        eda_means, eda_medians = [], []
        for condition in conditions:
            start_index, end_index = synchronized_times[participant][condition]
            start_index = int(start_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
            end_index = int(end_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
            _, filtered_signal, tonic_signal, phasic_signal = eda_dictionary[participant]
            signal_condition = filtered_signal[start_index:end_index]
            eda_means.append(np.mean(signal_condition))
            eda_medians.append(np.median(signal_condition))
        means_normalized = [((x - np.mean(eda_means)) / np.std(eda_means)) for x in eda_means]
        medians_normalized = [((x - np.mean(eda_medians)) / np.std(eda_medians)) for x in eda_medians]
        normalized_means_medians[participant] = {"means_norm": means_normalized, "medians_norm": medians_normalized}
    write_pickle("eda_normalized_means_medians.pickle", normalized_means_medians)
    return


# compute_normalized_means_medians()
# data = load_pickle("eda_normalized_means_medians.pickle")
# print(data)

# filtered_data = {}
#
# fig, ax = plt.subplots()

# for condition in [1]:
#     for participant in participants:
#         start_index, end_index = synchronized_times[participant][condition]
#         df = create_clean_dataframe_ecg_eda(participant).iloc[start_index:end_index]
#         eda_signal = np.abs(df["Sensor-C:SC/GSR"].values)
#         selected_values = eda_signal[::32]
#         filtered_eda_signal, tonic_signal, phasic_signal = filter_eda_data(selected_values,
#                                                                            1000,
#                                                                            1000,
#                                                                            200
#                                                                            )
#         mean_value = np.mean(filtered_eda_signal)
#         standard_deviation = np.std(filtered_eda_signal)
#         normalized_data = [((x - mean_value) / standard_deviation) for x in filtered_eda_signal]
#         filtered_data[participant] = [eda_signal, filtered_eda_signal, tonic_signal, phasic_signal]
#         plt.plot(normalized_data, label=f"participant {participant}")
#
# # write_pickle("eda_signal_filtered_normalized.pickle", filtered_data)
# ax.set_title("Skin Conductance (EDA)")
# ax.set_xlabel("Time (mm:ss)")
# ax.set_ylabel("Skin Conductance (micro-Siemens")
# plt.legend()
# plt.show()


