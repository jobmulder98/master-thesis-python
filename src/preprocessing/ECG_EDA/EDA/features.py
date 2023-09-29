from dotenv import load_dotenv
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd

from src.preprocessing.ECG_EDA.clean_raw_data import create_clean_dataframe

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def eda_features(participant_number, start_index: int, end_index: int, plot=False) -> dict:
    dataframe = create_clean_dataframe(participant_number)
    eda_signal = dataframe["Sensor-C:SC/GSR"].iloc[start_index:end_index].values
    signals, info = nk.eda_process(eda_signal, sampling_rate=ECG_SAMPLE_RATE)
    scl_signal = signals["EDA_Tonic"]
    scr_peaks = signals["SCR_Peaks"]

    scr_amplitude_all = signals["SCR_Amplitude"].values
    scr_amplitude_indexes = np.nonzero(scr_amplitude_all)
    scr_amplitude_peaks = scr_amplitude_all[scr_amplitude_indexes]

    scr_rise_time_all = signals["SCR_RiseTime"].values
    scr_rise_time_indexes = np.nonzero(scr_rise_time_all)
    scr_rise_time_peaks = scr_rise_time_all[scr_rise_time_indexes]

    features = {
        "Mean SCL": np.mean(scl_signal),
        "Mean derivative SCL": np.mean(np.gradient(scl_signal)),
        "Number of SCR peaks": np.count_nonzero(scr_peaks == 1),
        "Mean SCR peak amplitude": np.mean(scr_amplitude_peaks),
        "Mean rise time SCR peaks": np.mean(scr_rise_time_peaks),
    }

    if plot:
        nk.eda_plot(signals, info)
        plt.show()

    return features


participant_no = 101
start_index_eda, end_index_eda = 0, -1

print("EDA features:")
for k, v in eda_features(participant_no, start_index_eda, end_index_eda, plot=True).items():
    print("- %s: %s" % (k, v))
