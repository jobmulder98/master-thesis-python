import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import os
from scipy.stats import zscore
from dotenv import load_dotenv

from src.preprocessing.ECG_EDA.clean_raw_data import (
    create_clean_dataframe,
    synchronize_all_conditions,
    text_to_dataframe,
)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def mean_hr(dataframe, start_index, end_index):
    return dataframe["[B] Heart Rate"].iloc[start_index:end_index].mean()


def mean_hrv_amplitude(dataframe, start_index, end_index):
    return dataframe["[B] HRV Amp."].iloc[start_index:end_index].mean()


def detect_r_peaks(dataframe, sample_frequency=ECG_SAMPLE_RATE, plot=False):
    filtered_ecg_signal = dataframe["Sensor-B:EEG-Filtered"].values
    r_peaks_uncorrected = ecg.ecg(filtered_ecg_signal,
                                  sampling_rate=sample_frequency,
                                  show=False)[2]
    r_peaks_corrected = ecg.correct_rpeaks(filtered_ecg_signal,
                                           r_peaks_uncorrected,
                                           sampling_rate=sample_frequency,
                                           tol=0.05)[0]
    if plot:
        plt.plot(filtered_ecg_signal)
        plt.plot(r_peaks_corrected, filtered_ecg_signal[r_peaks_corrected], 'ro')
        plt.title("Found R-Peaks")
        plt.xlabel("Amplitude (arbitrary unit)")
        plt.ylabel("Frame (1024 sample frequency)")
        plt.show()
    return r_peaks_corrected


def compute_rr_intervals(r_peaks, plot=False):
    rr_intervals = np.diff(r_peaks)
    rr_corrected = rr_intervals.copy()
    rr_corrected[np.abs(zscore(rr_intervals)) > 2] = np.median(rr_intervals)
    if plot:
        plt.figure(figsize=(20, 7))
        plt.title("RR-intervals")
        plt.xlabel("Time (ms)")
        plt.ylabel("RR-interval (ms)")
        plt.plot(rr_intervals, color="red", label="RR-intervals")
        plt.plot(rr_corrected, color="green", label="RR-intervals after correction")
        plt.legend()
        plt.show()
    return rr_corrected


def time_domain_features(rr):
    features = {}
    hr = 60000 / rr
    features['Mean RR (ms)'] = np.mean(rr)
    features['STD RR/SDNN (ms)'] = np.std(rr)
    features['Mean HR (beats/min)'] = np.mean(hr)
    features['STD HR (beats/min)'] = np.std(hr)
    features['Min HR (beats/min)'] = np.min(hr)
    features['Max HR (beats/min)'] = np.max(hr)
    features['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))
    features['NNxx'] = np.sum(np.abs(np.diff(rr)) > 50) * 1
    features['pNNxx (%)'] = 100 * np.sum((np.abs(np.diff(rr)) > 50) * 1) / len(rr)
    return features


df = create_clean_dataframe(101)
r_peaks = detect_r_peaks(df, plot=False)
rr_intervals = compute_rr_intervals(r_peaks, plot=True)


# condition_indexes = synchronize_all_conditions(102)
# for key, value in condition_indexes.items():
#     print(f"The mean heart rate for condition {key} is {mean_hr(clean_dataframe, value[0], value[1])}")
# plt.plot(clean_dataframe["Sensor-B:EEG"])
# plt.show()
