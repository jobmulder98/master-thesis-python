import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema, lfilter


from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.ecg_eda.eda.filtering import (
    replace_values_above_threshold_to_nan,
    add_margin_around_nan_values,
)
from src.preprocessing.hmd.movements.filtering_movements import (
    filter_head_movement_data,
    filter_hand_movement_data,
)


def head_stillness(participant, condition, threshold=3):
    df = create_clean_dataframe_hmd(participant, condition)
    times, acc = filter_head_movement_data(df)
    plt.plot(times, acc)
    plt.show()
    dt = times[1] - times[0]
    head_stillness_time = 0
    for i in range(len(acc)):
        if abs(acc[i]) < threshold:
            head_stillness_time += dt
    return head_stillness_time


def head_movement_peaks(signal, threshold: float):
    peaks = argrelextrema(signal, np.greater_equal, order=3)[0]
    peaks_corrected = []
    for peak in peaks:
        if signal[peak] > threshold:
            peaks_corrected.append(peak)
    return peaks_corrected


def calculate_acceleration(dataframe, plot=False):
    dataframe["hmdAngularVelocity"] = dataframe["hmdEuler"].diff() / dataframe["deltaSeconds"]
    dataframe["hmdAngularAcceleration"] = dataframe["hmdAngularVelocity"].diff() / dataframe["deltaSeconds"]

    dataframe["hmdAngularVelocity"].fillna(0, inplace=True)
    dataframe["hmdAngularAcceleration"].fillna(0, inplace=True)

    dataframe['hmdAngularVelocityVectorize'] = dataframe['hmdAngularVelocity'].apply(lambda acc: np.linalg.norm(acc))
    dataframe['hmdAngularAccelerationVectorize'] = dataframe['hmdAngularAcceleration'].apply(lambda acc: np.linalg.norm(acc))

    dataframe['hmdAngularVelocityVectorize10'] = dataframe['hmdAngularVelocityVectorize'].rolling(10).mean()
    dataframe['hmdAngularAccelerationVectorize10'] = dataframe['hmdAngularAccelerationVectorize'].rolling(10).mean()

    hm_peaks = head_movement_peaks(dataframe.hmdAngularAccelerationVectorize10.values, 250)

    if plot:
        plt.plot(dataframe["hmdAngularAccelerationVectorize10"], color="blue")
        custom_xticks = np.arange(0, len(dataframe["hmdAngularAccelerationVectorize10"]),
                                  step=len(dataframe["hmdAngularAccelerationVectorize10"]) / 12)
        custom_xlabel_positions = np.arange(0, 120, step=10)
        custom_xlabels = ['{:.1f}'.format(x) for x in custom_xlabel_positions]
        plt.xticks(custom_xticks, custom_xlabels, rotation=45)
        dataframe.iloc[hm_peaks].hmdAngularAccelerationVectorize10.plot(style='.', lw=10, color='red', marker=".")
        plt.show()
    return


def head_movement_features(dataframe: pd.DataFrame, plot=False) -> dict:
    calculate_acceleration(dataframe, plot=plot)
    features = {
        "mean head acceleration": np.mean(dataframe['hmdAngularAccelerationVectorize']),
        "min head acceleration": np.min(dataframe['hmdAngularAccelerationVectorize']),
        "max head acceleration": np.max(dataframe['hmdAngularAccelerationVectorize']),
    }
    return features


# participant_number = 1
# head_stillness(participant_number, 1)
# condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]
# for condition in np.arange(1, 8):
#     print(f"Total head stillness time for condition {condition_names[condition-1]}: {head_stillness(participant_number, condition)}")
