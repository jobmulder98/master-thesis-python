import biosppy.signals.ecg
from dotenv import load_dotenv
import os
import pandas as pd
import biosppy
from pyhrv.hrv import hrv
import warnings

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
                 start_index: int,
                 end_index: int,
                 plot_biosppy_analysis=False,
                 plot_hrv_analysis=False,
                 plot_ecg=False,
                 plot_tachogram=False) -> dict:
    """
    For documentation see https://pyhrv.readthedocs.io/en/latest/index.html
    """

    raw_ecg_signal = dataframe["Sensor-B:EEG"].iloc[start_index:end_index].values
    t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(raw_ecg_signal,
                                                         sampling_rate=ECG_SAMPLE_RATE,
                                                         show=plot_biosppy_analysis)[:3]
    signal_features = hrv(rpeaks=t[rpeaks], show=plot_hrv_analysis, plot_ecg=plot_ecg, plot_tachogram=plot_tachogram)
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
    return features


# participant_no = 5
# condition = 7
# sync = {5: {1: [2592486, 2717829],
#             2: [250177, 375555],
#             3: [1022819, 1148211],
#             4: [2278270, 2403647],
#             5: [695468, 820912],
#             6: [1418978, 1544377],
#             7: [1912605, 2038021]
#             }
#         }
# start_index_condition, end_index_condition = sync[participant_no][condition]
# df = create_clean_dataframe_ecg_eda(participant_no)
# print(ecg_features(
#     dataframe=df,
#     start_index=start_index_condition,
#     end_index=end_index_condition,
#     plot_biosppy_analysis=False,
#     plot_hrv_analysis=False,
#     plot_ecg=False,
#     plot_tachogram=False)
# )
