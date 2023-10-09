import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema, lfilter


from src.preprocessing.HMD.clean_raw_data import create_clean_dataframe_hmd

#  TODO: CHECK THE REST OF THE FEATURES AND DECIDE WHICH ONES ARE USEFUL + IMPLEMENT


def calculate_angular_velocity_acceleration(dataframe, plot=False):
    dataframe["hmdAngularVelocity"] = dataframe["hmdEuler"].diff() / dataframe["deltaSeconds"]
    dataframe["hmdAngularVelocity"].fillna(0, inplace=True)
    dataframe['hmdAngularVelocityNorm'] = dataframe['hmdAngularVelocity'].apply(lambda acc: np.linalg.norm(acc))
    dataframe['hmdAngularVelocityNorm10'] = dataframe['hmdAngularVelocityNorm'].rolling(10).mean()

    dataframe["hmdAngularAcceleration"] = dataframe["hmdAngularVelocity"].diff() / dataframe["deltaSeconds"]
    dataframe["hmdAngularAcceleration"].fillna(0, inplace=True)
    dataframe['hmdAngularAccelerationNorm'] = dataframe['hmdAngularAcceleration'].apply(lambda acc: np.linalg.norm(acc))
    dataframe['hmdAngularAccelerationNorm10'] = dataframe['hmdAngularAccelerationNorm'].rolling(10).mean()

    hm_peaks = head_movement_peaks(dataframe, 18)

    if plot:
        plt.plot(df["hmdAngularAccelerationNorm10"], color="blue")
        custom_xticks = np.arange(0, len(df["hmdAngularAccelerationNorm10"]), step=len(df["hmdAngularAccelerationNorm10"]) / 12)
        custom_xlabel_positions = np.arange(0, 120, step=10)  # Custom x-label positions
        custom_xlabels = ['{:.1f}'.format(x) for x in custom_xlabel_positions]  # Custom x-labels
        plt.xticks(custom_xticks, custom_xlabels, rotation=45)
        plt.axhline(15, color="red")
        df.iloc[hm_peaks].hmdAngularAccelerationNorm10.plot(style='.', lw=10, color='red', marker=".")
        plt.show()
    return


def head_movement_peaks(dataframe: pd.DataFrame, threshold: float):
    peaks = argrelextrema(dataframe.hmdAngularAccelerationNorm10.values, np.greater_equal, order=3)[0]
    peaks_corrected = []
    for peak in peaks:
        if dataframe["hmdAngularAccelerationNorm10"][peak] > threshold:
            peaks_corrected.append(peak)
    return peaks_corrected


def head_movement_features(dataframe: pd.DataFrame, plot=False) -> dict:
    calculate_angular_velocity_acceleration(dataframe, plot=plot)
    features = {
        "mean head acceleration": np.mean(dataframe['hmdAngularAccelerationNorm']),
        "min head acceleration": np.min(dataframe['hmdAngularAccelerationNorm']),
        "max head acceleration": np.max(dataframe['hmdAngularAccelerationNorm']),
    }
    return features


# participant_number = 102
# condition = 1
# start_index = 0
# end_index = -1
# print(head_movement_features(participant_number, condition, start_index, end_index))
