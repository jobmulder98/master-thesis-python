import numpy as np
from dotenv import load_dotenv
import os
import pickle
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.ndimage import median_filter

from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda
from src.preprocessing.helper_functions.general_helpers import (format_time,
                                                                interpolate_nan_values,
                                                                write_pickle,
                                                                load_pickle)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
EDA_SAMPLE_RATE = int(os.getenv("EDA_SAMPLE_RATE"))


def replace_values_above_threshold_to_nan(signal, threshold):
    filtered_signal = np.where(signal <= threshold, signal, np.nan)
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


participants = np.arange(1, 23)
conditions = np.arange(1, 8)

synchronized_times = load_pickle("synchronized_times.pickle")

filtered_data = {}

for participant in participants:
    df = create_clean_dataframe_ecg_eda(participant)  #.iloc[start_index_condition:end_index_condition]
    eda_signal = np.abs(df["Sensor-C:SC/GSR"].values)
    selected_values = eda_signal[::32]
    filtered_eda_signal, tonic_signal, phasic_signal = filter_eda_data(eda_signal,
                                                                       1000,
                                                                       1000,
                                                                       200
                                                                       )

    # mean_value = np.mean(filtered_eda_signal)
    # standard_deviation = np.std(filtered_eda_signal)
    # normalized_data = [((x - mean_value) / standard_deviation) for x in filtered_eda_signal]
    # for condition in conditions:
    #     start_index_condition, end_index_condition = synchronized_times[participant][condition]
    filtered_data[participant] = [eda_signal, filtered_eda_signal, tonic_signal, phasic_signal]
    # plt.plot(normalized_data)

write_pickle("eda_signal_filtered_normalized.pickle", filtered_data)
#
# plt.show()
# fig, ax = plt.subplots()
# time = np.arange(len(eda_signal)) / ECG_SAMPLE_RATE
# ax.plot(time, normalized_data)
# ax.set_title("Skin Conductance (EDA)")
# ax.set_xlabel("Time (mm:ss)")
# ax.set_ylabel("Skin Conductance (micro-Siemens")
# ax.xaxis.set_major_formatter(FuncFormatter(format_time))
# plt.show()
