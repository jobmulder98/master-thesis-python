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
    obtain_filtered_signal_and_peaks,
)

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


