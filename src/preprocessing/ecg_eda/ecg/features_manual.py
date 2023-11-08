import numpy as np
import pandas as pd
from biosppy.signals import ecg
from matplotlib import pyplot as plt
from scipy.stats import zscore

from preprocessing.ecg_eda.ecg.features import ECG_SAMPLE_RATE


"""
This file could be used to understand the ECG features.

Also plots could be made to see the behavior of the ECG signal 

The other features file is used, since the biosppy.signals in combination with the pyhrv.hrv 
package since this is a more powerful calculator. 
"""


def detect_r_peaks(dataframe: pd.DataFrame,
                   start_index: int,
                   end_index: int,
                   sample_frequency=ECG_SAMPLE_RATE,
                   plot=False):
    filtered_ecg_signal = dataframe["Sensor-B:EEG-Filtered"].iloc[start_index:end_index].values
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


def ecg_features(dataframe: pd.DataFrame,
                 start_index: int,
                 end_index: int,
                 plot_r_peaks=False,
                 plot_rr_intervals=False) -> dict:
    """
    Manual determination of the ECG features with biosppy.

    There are some performance differences, elected is now pyHRV (see uncommented ecg_features function)
    """
    r_peaks = detect_r_peaks(dataframe, start_index, end_index, plot=plot_r_peaks)
    rr_intervals = compute_rr_intervals(r_peaks, plot=plot_rr_intervals)
    hr = 60 / (rr_intervals / ECG_SAMPLE_RATE)
    mean_hr = (len(rr_intervals) - 1) / ((r_peaks[-1] - r_peaks[0]) / ECG_SAMPLE_RATE) * 60
    low_frequency_signal = dataframe["[B] HRV-LF Power (0.04-0.16 Hz)"].iloc[start_index:end_index]
    high_frequency_signal = dataframe["[B] HRV-HF Power (0.16-0.4 Hz)"].iloc[start_index:end_index]
    lf_power = low_frequency_signal.sum() / ECG_SAMPLE_RATE
    hf_power = high_frequency_signal.sum() / ECG_SAMPLE_RATE
    total_power = lf_power + hf_power

    features = {
        "Mean RR (ms)": np.mean(rr_intervals),
        "STD RR/SDNN (ms)": np.std(rr_intervals),
        "Mean HR (beats/min)": mean_hr,
        "STD HR (beats/min)": np.std(hr),
        "Min HR (beats/min)": np.min(hr),
        "Max HR (beats/min)": np.max(hr),
        "RMSSD (ms)": np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
        "NN50": np.sum(np.abs(np.diff(rr_intervals)) > 50) * 1,
        "pNN50 (%)": 100 * np.sum((np.abs(np.diff(rr_intervals)) > 50) * 1) / len(rr_intervals),
        "Power LF (ms2)": lf_power,
        "Power HF (ms2)": hf_power,
        "Power Total (ms2)": total_power,
        "LF/HF": (lf_power / hf_power),
        "Peak LF (Hz)": np.max(low_frequency_signal),
        "Peak HF (Hz)": np.max(high_frequency_signal),
    }
    return features
