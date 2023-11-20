import biosppy.signals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import typing as npt
import pickle
from dotenv import load_dotenv
import os

from preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda


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


def correct_rpeaks_manually(participant, condition, corrected_rpeaks: npt.NDArray) -> npt.NDArray:
    if participant == 2 and condition == 7:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 70182))
        corrected_rpeaks = np.append(corrected_rpeaks, 70462)
        corrected_rpeaks = np.append(corrected_rpeaks, 71279)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 7 and condition == 7:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 42024))
        corrected_rpeaks = np.append(corrected_rpeaks, 42262)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 11 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 10176))
        corrected_rpeaks = np.append(corrected_rpeaks, 9996)
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 10789))
        corrected_rpeaks = np.append(corrected_rpeaks, 10698)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 11 and condition == 5:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 5491))
        corrected_rpeaks = np.append(corrected_rpeaks, 5303)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 15 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 11942))
        corrected_rpeaks = np.append(corrected_rpeaks, 11788)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 15 and condition == 4:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 21924))
        corrected_rpeaks = np.append(corrected_rpeaks, 21722)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 16 and condition == 6:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 16540))
        corrected_rpeaks = np.append(corrected_rpeaks, 16315)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 21 and condition == 1:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 84158))
    if participant == 21 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 108456))
    if participant == 21 and condition == 5:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 5074))
        corrected_rpeaks = np.append(corrected_rpeaks, 5003)
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 101358))
        corrected_rpeaks = np.append(corrected_rpeaks, 101217)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 21 and condition == 6:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 28986))
        corrected_rpeaks = np.append(corrected_rpeaks, 28850)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 21 and condition == 7:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 51707))
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 80818))
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 81422))
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 109055))
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 109560))
        corrected_rpeaks = np.append(corrected_rpeaks, 51858)
        corrected_rpeaks = np.append(corrected_rpeaks, 80656)
        corrected_rpeaks = np.append(corrected_rpeaks, 109321)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 22 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 56831))
    if participant == 22 and condition == 4:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 114480))
        corrected_rpeaks = np.append(corrected_rpeaks, 114440)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 22 and condition == 7:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 44485))
        corrected_rpeaks = np.append(corrected_rpeaks, 44339)
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 45469))
        corrected_rpeaks = np.append(corrected_rpeaks, 45593)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    return corrected_rpeaks


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


participants = np.arange(5, 6)
with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
    synchronized_times = pickle.load(handle)

for participant_no in participants:
    for condition in np.arange(7, 8):
        start_index_condition, end_index_condition = synchronized_times[participant_no][condition]
        df = create_clean_dataframe_ecg_eda(participant_no)
        _, _, rpeaks = (obtain_filtered_signal_and_peaks(
            dataframe=df,
            participant=participant_no,
            condition=condition,
            start_index=start_index_condition,
            end_index=end_index_condition,
            plot_biosppy_analysis=False)
        )
        filtered_rpeaks = delete_outliers_iqr(rpeaks)


# plt.plot(60 / (np.diff(rpeaks) / ECG_SAMPLE_RATE))
time_axis = np.cumsum(np.diff(rpeaks)) / ECG_SAMPLE_RATE
plt.plot(time_axis, filtered_rpeaks, marker=".")
plt.show()

