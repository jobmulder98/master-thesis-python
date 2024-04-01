import biosppy.signals
import numpy as np
import pandas as pd
from numpy import typing as npt
from dotenv import load_dotenv
import os

from preprocessing.helper_functions.general_helpers import interpolate_nan_values, load_pickle, write_pickle


load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def check_for_corrupted_data(participant: int, condition: int, signal: npt.NDArray) -> npt.NDArray:
    if participant == 22 and condition == 6:
        signal = signal[:55000]
    return signal


def delete_outliers_iqr(detected_rr_peaks):
    rr_intervals = np.diff(detected_rr_peaks)
    heart_rate = 60 / (rr_intervals / ECG_SAMPLE_RATE)
    sorted_data = np.sort(heart_rate)
    quartile_1 = sorted_data[int(len(heart_rate) * 0.25)]
    quartile_3 = sorted_data[int(len(heart_rate) * 0.75)]
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - 1.5 * iqr
    upper_bound = quartile_3 + 1.5 * iqr
    filtered_heart_rate = [x if lower_bound <= x <= upper_bound else np.nan for x in heart_rate]
    return filtered_heart_rate


def obtain_filtered_signal_and_peaks(dataframe: pd.DataFrame,
                                     participant: int,
                                     condition: int,
                                     start_index: int,
                                     end_index: int,
                                     plot_biosppy_analysis=False):
    raw_ecg_signal = dataframe["Sensor-B:EEG"].iloc[start_index:end_index].values
    raw_ecg_signal = check_for_corrupted_data(participant, condition, raw_ecg_signal)
    t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(raw_ecg_signal,
                                                         sampling_rate=ECG_SAMPLE_RATE,
                                                         show=plot_biosppy_analysis)[:3]
    return t, filtered_signal, rpeaks


def calculate_mean_heart_rate(times, rpeaks):
    if rpeaks is None:
        return None
    number_of_intervals = len(rpeaks)
    total_time = times[-1] - times[0]
    return number_of_intervals / (total_time / 60)


def calculate_heart_rate_variability(rpeaks):
    if rpeaks is None:
        return None
    interpolated_rpeaks = interpolate_nan_values(rpeaks)
    return np.std(interpolated_rpeaks)


def calculate_rmssd(rpeaks):
    if rpeaks is None:
        return None
    rpeaks_interpolated = interpolate_nan_values(rpeaks)
    rr_intervals = [60 / (hr / ECG_SAMPLE_RATE) for hr in rpeaks_interpolated]
    differences_between_consecutive_intervals = np.diff(rr_intervals)
    squared_diff = differences_between_consecutive_intervals ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmssd = np.sqrt(mean_squared_diff)
    return rmssd
