from dotenv import load_dotenv
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import pickle

from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda
from src.preprocessing.ecg_eda.eda.signal_correction import filter_eda_signal, plot_filtered_signals

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def eda_features(dataframe: pd.DataFrame,
                 participant: int,
                 condition: int,
                 start_index: int,
                 end_index: int,
                 plot_eda_analysis=False,
                 plot_signals=False) -> dict:

    eda_signal = dataframe["Sensor-C:SC/GSR"]
    filtered_signal = filter_eda_signal(eda_signal)
    filtered_signal_condition = filtered_signal[start_index:end_index]

    if plot_signals:
        plot_filtered_signals(np.abs(eda_signal[start_index:end_index]), filtered_signal_condition)

    signals, info = nk.eda_process(filtered_signal_condition, sampling_rate=ECG_SAMPLE_RATE)
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

    if plot_eda_analysis:
        nk.eda_plot(signals, info)
        plt.show()

    return features


participant_no = 21
with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
    synchronized_times = pickle.load(handle)
# condition = 7

for condition in np.arange(1, 8):
    start_index_condition, end_index_condition = synchronized_times[participant_no][condition]
    df = create_clean_dataframe_ecg_eda(participant_no)
    print(eda_features(
        dataframe=df,
        participant=participant_no,
        condition=condition,
        start_index=start_index_condition,
        end_index=end_index_condition,
        plot_eda_analysis=False,
        plot_signals=True)
    )
