import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
import warnings
import matplotlib.pyplot as plt

from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def remove_spikes(raw_signal):
    filtered_signal = raw_signal
    time_in_seconds = np.arange(len(raw_signal))
    time_in_seconds_divided = time_in_seconds / 1024
    plt.plot(time_in_seconds_divided, raw_signal)
    plt.show()
    return filtered_signal


def clip_data(unclipped, high_clip, low_clip):
    np_unclipped = np.array(unclipped)
    cond_high_clip = (np_unclipped > HIGH_CLIP) | (np_unclipped < LOW_CLIP)
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


DELTA = 0.1
HIGH_CLIP = 1000
LOW_CLIP = -1000
RAND_HIGH = 0.98
RAND_LOW = 0.02
SPAN = 10
SPIKE = 2

participants = np.arange(20, 23)
with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
    synchronized_times = pickle.load(handle)
condition = 2
# start_index_condition, end_index_condition = synchronized_times[participant_no][condition]
start_index_condition, end_index_condition = 0, -1
for participant in participants:
    df = create_clean_dataframe_ecg_eda(participant)
    eda_signal = df["Sensor-C:SC/GSR"].iloc[start_index_condition:end_index_condition].values
    remove_spikes(eda_signal)


# signal_clipped = clip_data(eda_signal, HIGH_CLIP, LOW_CLIP)
# signal_ewma = ewma_fb(signal_clipped, SPAN)
# signal_without_outliers = remove_outliers(signal_clipped, signal_ewma, DELTA)
# signal_corrected = np.interpolate(signal_without_outliers)
#
# plt.plot(eda_signal, color="blue")
# plt.plot(signal_corrected, color="red")
# plt.show()

#  TODO: Kan niet gebruikt worden: [1, 2, 6 (signaal blijft rond de 1 hangen?), 8, 9, 21 (einde is noise), 22 (einde is noise)]
#  TODO: Kan wel gebruikt worden: [3, 4, 5, 7, 10 (lage pieken), 11 (lage pieken) , 12 (lage pieken), 13, 14,
#   15 (negatief signaal), 16, 17, 18 (lage pieken), 20]
#  TODO: Twijfel of gebruikt kan worden: [19 (hele hoge skin conductance (rond de 600) ]

#  TODO: synchroniseer SC met conditions en kijk of pieken vooral zijn tijdens condities door beweging. Gebruik 13 of 20.