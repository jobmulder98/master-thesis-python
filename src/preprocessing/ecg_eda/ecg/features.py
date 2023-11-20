import biosppy.signals.ecg
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import biosppy
import pickle
from pyhrv.hrv import hrv
import warnings
import matplotlib.pyplot as plt

from preprocessing.ecg_eda.ecg.filtering import (
    correct_rpeaks,
    correct_rpeaks_manually,
    check_for_corrupted_data,
    obtain_filtered_signal_and_peaks,
)
from preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning,
                        message="Signal duration is to short for segmentation into 300000s.")
warnings.filterwarnings("ignore", category=UserWarning, message="Signal duration too short for SDANN computation.")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="CAUTION: The TINN computation is currently providing incorrect results in the most "
                                "cases due to a malfunction of the function.")

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def ecg_features(dataframe: pd.DataFrame,
                 participant: int,
                 condition: int,
                 start_index: int,
                 end_index: int,
                 plot_corrected_peaks=False,
                 plot_biosppy_analysis=False,
                 plot_hrv_analysis=False,
                 plot_ecg=False,
                 plot_tachogram=False
                 ):
    """
    For documentation see https://pyhrv.readthedocs.io/en/latest/index.html
    """
    t, filtered_signal, rpeaks = obtain_filtered_signal_and_peaks(dataframe,
                                                                  participant,
                                                                  condition,
                                                                  start_index,
                                                                  end_index)
    corrected_peaks = correct_rpeaks(rpeaks)
    corrected_peaks = correct_rpeaks_manually(participant, condition, corrected_peaks)
    corrected_peaks = np.unique(corrected_peaks)
    # print(corrected_peaks)

    if plot_corrected_peaks:
        fig, ax = plt.subplots()
        ax.plot(filtered_signal)
        ax.vlines(corrected_peaks, ymin=min(filtered_signal), ymax=max(filtered_signal), color='red')
        ax.vlines(rpeaks, ymin=min(filtered_signal), ymax=max(filtered_signal), linestyles='dashed', color='green')
        plt.show()

    signal_features = hrv(rpeaks=t[corrected_peaks], show=plot_hrv_analysis, plot_ecg=plot_ecg,
                          plot_tachogram=plot_tachogram)
    features = {
        "Mean NNI (ms)": signal_features["nni_mean"],
        "Minimum NNI": signal_features["nni_min"],
        "Maximum NNI": signal_features["nni_max"],
        "Mean HR (beats/min)": signal_features["hr_mean"],
        "STD HR (beats/min)": signal_features["hr_std"],
        "Min HR (beats/min)": signal_features["hr_min"],
        "Max HR (beats/min)": signal_features["hr_max"],
        "SDNN (ms)": signal_features["sdnn"],
        "RMSSD (ms)": signal_features["rmssd"],
        "NN50": signal_features["nn50"],
        "pNN50 (%)": signal_features["pnn50"],
        "Power VLF (ms2)": signal_features["fft_abs"][0],
        "Power LF (ms2)": signal_features["fft_abs"][1],
        "Power HF (ms2)": signal_features["fft_abs"][2],
        "Power Total (ms2)": signal_features["fft_total"],
        "LF/HF": signal_features["fft_ratio"],
        "Peak VLF (Hz)": signal_features["fft_peak"][0],
        "Peak LF (Hz)": signal_features["fft_peak"][1],
        "Peak HF (Hz)": signal_features["fft_peak"][2],
    }
    plt.close('all')  # avoid error: Failed to allocate bitmap
    plt.clf()  # avoid error: Failed to allocate bitmap
    return features


# participants = np.arange(1, 23)
# with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
#     synchronized_times = pickle.load(handle)
#
# for participant_no in participants:
#     for condition in np.arange(7, 8):
#         start_index_condition, end_index_condition = synchronized_times[participant_no][condition]
#         df = create_clean_dataframe_ecg_eda(participant_no)
#         print(ecg_features(
#             dataframe=df,
#             participant=participant_no,
#             condition=condition,
#             start_index=start_index_condition,
#             end_index=end_index_condition,
#             plot_corrected_peaks=False,
#             plot_biosppy_analysis=False,
#             plot_hrv_analysis=False,
#             plot_ecg=False,
#             plot_tachogram=False, ))

