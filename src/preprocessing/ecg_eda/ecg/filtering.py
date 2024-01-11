import biosppy.signals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import typing as npt
import pickle
from dotenv import load_dotenv
import os

from preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda
from preprocessing.helper_functions.general_helpers import interpolate_nan_values, load_pickle, write_pickle


load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))



def correct_rpeaks(peak_indices):
    peak_indices_diff = np.diff(peak_indices)
    threshold = 200
    corrected_peaks_indices = [peak_indices_diff[0]]
    skip_one_iteration = False
    for i in range(1, len(peak_indices_diff) - 1):
        if not skip_one_iteration:
            previous_peak = np.mean(corrected_peaks_indices)
            current_peak = peak_indices_diff[i]
            next_peak = peak_indices_diff[i+1]
            difference = np.abs(current_peak - previous_peak)
            if difference >= threshold:
                if (current_peak + next_peak - previous_peak) <= threshold:
                    corrected_peaks_indices.append(current_peak + next_peak)
                    skip_one_iteration = True
                elif np.abs(current_peak - 2*next_peak) <= threshold:
                    corrected_peaks_indices.append(current_peak // 2)
                    if current_peak % 2 == 0:
                        corrected_peaks_indices.append(current_peak // 2)
                    else:
                        corrected_peaks_indices.append((current_peak // 2) + 1)
                elif current_peak + next_peak - 2*previous_peak <= threshold:
                    corrected_peaks_indices.append((current_peak + next_peak) // 2)
                    corrected_peaks_indices.append((current_peak + next_peak) // 2)  # Do this two times
                    skip_one_iteration = True
                else:
                    corrected_peaks_indices.append(current_peak)
            else:
                corrected_peaks_indices.append(current_peak)
        else:
            skip_one_iteration = False
    corrected_peaks_indices.append(peak_indices_diff[-1])
    corrected_peaks_indices.insert(0, peak_indices[0])
    return np.cumsum(corrected_peaks_indices)


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


# participants = np.arange(1, 22)
# with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
#     synchronized_times = pickle.load(handle)
#
# unfiltered_times_of_rpeaks_all = {}
# unfiltered_rpeaks_all = {}
# mask_all = {}
# filtered_times_of_rpeaks_all = {}
# filtered_rpeaks_all = {}

# for condition in np.arange(1, 8):
#     times = []
#     heart_rate_values = []
#     masks = []
#     for participant_no in participants:
#         if participant_no == 12 and condition == 4:
#             times.append(None)
#             heart_rate_values.append(None)
#         else:
#             start_index_condition, end_index_condition = synchronized_times[participant_no][condition]
#             df = create_clean_dataframe_ecg_eda(participant_no)
#             t, _, rpeaks = (obtain_filtered_signal_and_peaks(
#                 dataframe=df,
#                 participant=participant_no,
#                 condition=condition,
#                 start_index=start_index_condition,
#                 end_index=end_index_condition,
#                 plot_biosppy_analysis=False)
#             )
#             time_filtered_rpeaks = np.array(t[rpeaks][1:]).astype(np.double)
#             filtered_rpeaks = np.array(delete_outliers_iqr(rpeaks)).astype(np.double)
#             times.append(time_filtered_rpeaks)
#             heart_rate_values.append(filtered_rpeaks)
#             mask = np.isfinite(filtered_rpeaks)
#             masks.append(mask)
#             # times.append(time_filtered_rpeaks[mask])
#             # heart_rate_values.append(filtered_rpeaks[mask])
#     filtered_times_of_rpeaks_all[condition] = times
#     filtered_rpeaks_all[condition] = heart_rate_values
#     mask_all[condition] = masks


###### HEART RATE AND VARIABILITY FOR PARTICIPANT 22 #######
# hr, hrv = [], []
# for condition in np.arange(1, 8):
#     participant_no = 22
#     times = []
#     heart_rate_values = []
#     masks = []
#     start_index_condition, end_index_condition = synchronized_times[participant_no][condition]
#     df = create_clean_dataframe_ecg_eda(participant_no)
#     t, _, rpeaks = (obtain_filtered_signal_and_peaks(
#         dataframe=df,
#         participant=participant_no,
#         condition=condition,
#         start_index=start_index_condition,
#         end_index=end_index_condition,
#         plot_biosppy_analysis=False)
#     )
#     time_filtered_rpeaks = np.array(t[rpeaks][1:]).astype(np.double)
#     filtered_rpeaks = np.array(delete_outliers_iqr(rpeaks)).astype(np.double)
#     hr.append(calculate_mean_heart_rate(time_filtered_rpeaks, filtered_rpeaks))
#     hrv.append(calculate_rmssd(filtered_rpeaks))
#
# print(hr)
# print(hrv)
# time_and_rpeaks = [filtered_times_of_rpeaks_all, filtered_rpeaks_all, mask_all]
# write_pickle("ecg_data_unfiltered", time_and_rpeaks)


