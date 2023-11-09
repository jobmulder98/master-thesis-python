from dotenv import load_dotenv
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd

from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda
from src.preprocessing.helper_functions.general_helpers import butter_lowpass_filter

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def eda_features(dataframe: pd.DataFrame, start_index: int, end_index: int, plot=False) -> dict:
    eda_signal = dataframe["Sensor-C:SC/GSR"].iloc[start_index:end_index].values
    filtered_data = butter_lowpass_filter(eda_signal)
    plt.plot(filtered_data)
    signals, info = nk.eda_process(filtered_data, sampling_rate=ECG_SAMPLE_RATE)
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


participant_no = 18
start_index_eda, end_index_eda = 0, -1
df = create_clean_dataframe_ecg_eda(participant_no)

# print("eda features:")
# for k, v in eda_features(df, start_index_eda, end_index_eda, plot=True).items():
#     print("- %s: %s" % (k, v))


participants = np.arange(3, 4)
for p in participants:
    df = create_clean_dataframe_ecg_eda(p)
    eda_features(df, start_index_eda, end_index_eda, plot=False)

plt.show()
