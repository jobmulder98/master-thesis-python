import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda
from src.preprocessing.helper_functions.general_helpers import format_time, butter_lowpass_filter

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def remove_spikes(signal: list) -> list:
    delta = 0.1
    span = 10
    signal_median = np.median(signal)
    dataframe = pd.DataFrame()
    dataframe["raw_signal"] = signal
    dataframe['y_clipped'] = clip_data(dataframe["raw_signal"].tolist(), signal_median + signal_median * 3, 0)
    dataframe['y_ewma_fb'] = ewma_fb(dataframe['y_clipped'], span)
    dataframe['y_remove_outliers'] = remove_outliers(dataframe['y_clipped'].tolist(),
                                                     dataframe['y_ewma_fb'].tolist(),
                                                     delta)
    dataframe['y_interpolated'] = dataframe['y_remove_outliers'].interpolate(method='polynomial', order=2)
    return dataframe['y_interpolated'].values


def clip_data(unclipped, high_clip, low_clip):
    np_unclipped = np.array(unclipped)
    cond_high_clip = (np_unclipped > high_clip) | (np_unclipped < low_clip)
    np_clipped = np.where(cond_high_clip, np.nan, np_unclipped)
    return np_clipped.tolist()


def ewma_fb(df_column, span):
    fwd = pd.Series.ewm(df_column, span=span).mean()
    bwd = pd.Series.ewm(df_column[::-1],span=span).mean()
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma


def remove_outliers(spikey, fbewma, delta):
    np_spikey = np.array(spikey)
    np_fbewma = np.array(fbewma)
    cond_delta = (np.abs(np_spikey-np_fbewma) > delta)
    np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
    return np_remove_outliers


def filter_eda_signal(eda_signal):
    eda_signal = np.abs(eda_signal)
    spike_filtered_signal = remove_spikes(eda_signal)
    filtered_signal = butter_lowpass_filter(spike_filtered_signal)
    if np.isnan(filtered_signal).all():
        return spike_filtered_signal
    return filtered_signal


def plot_filtered_signals(eda_signal, filtered_signal):
    x = np.arange(len(eda_signal)) / ECG_SAMPLE_RATE
    fig, ax = plt.subplots()
    ax.plot(x, eda_signal, color="red")
    ax.plot(x, filtered_signal, color="green")
    ax.set_title("Skin Conductance (EDA)")
    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Skin Conductance (micro-Siemens")
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    plt.show()
    return None


# participant = 15
# with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
#     synchronized_times = pickle.load(handle)
# condition = 1
# start_index_condition, end_index_condition = synchronized_times[participant][condition]
# df = create_clean_dataframe_ecg_eda(participant)  #.iloc[start_index_condition:end_index_condition]
# eda_signal = np.abs(df["Sensor-C:SC/GSR"].values)
# filtered_signal =filter_eda_signal(eda_signal, plot_filtered_signals=True)[start_index_condition:end_index_condition]


#  TODO: Kan niet gebruikt worden: [1, 2, 6 (signaal blijft rond de 1 hangen?), 8, 9, 21 (einde is noise), 22 (einde is noise)]
#  TODO: Kan wel gebruikt worden: [3, 4, 5, 7, 10 (lage pieken), 11 (lage pieken) , 12 (lage pieken), 13, 14,
#   15 (negatief signaal), 16, 17, 18 (lage pieken), 20]
#  TODO: Twijfel of gebruikt kan worden: [19 (hele hoge skin conductance (rond de 600) ]
