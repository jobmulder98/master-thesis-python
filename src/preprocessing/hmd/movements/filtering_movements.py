import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.ecg_eda.eda.filtering import (
    replace_values_above_threshold_to_nan,
    add_margin_around_nan_values,
    interpolate_nan_values,
    median_filter,
)
from src.preprocessing.helper_functions.general_helpers import moving_average, butter_lowpass_filter
conditions = np.arange(1, 8)
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]


def time_derivative(dataframe: pd.DataFrame, x_columnname: str, x_dot_columnname: str):
    dataframe[x_dot_columnname] = dataframe[x_columnname].diff() / 0.05  #dataframe["deltaSeconds"]
    return dataframe


def filter_head_movement_data(dataframe: pd.DataFrame):
    dataframe = time_derivative(dataframe, "hmdEuler", "hmdAngularVelocity")
    dataframe = time_derivative(dataframe, "hmdAngularVelocity", "hmdAngularAcceleration")

    dataframe["hmdAngularVelocity"].fillna(0, inplace=True)
    dataframe["hmdAngularAcceleration"].fillna(0, inplace=True)

    dataframe["hmdAngularVelocityVectorize"] = dataframe["hmdAngularVelocity"].apply(lambda x: np.linalg.norm(x))
    dataframe["hmdAngularAccelerationVectorize"] = dataframe["hmdAngularAcceleration"].apply(lambda x: np.linalg.norm(x))

    head_movement_velocity = dataframe["hmdAngularVelocityVectorize"].values
    head_movement_velocity = replace_values_above_threshold_to_nan(head_movement_velocity, threshold=20)
    head_movement_velocity = add_margin_around_nan_values(head_movement_velocity, 500)
    head_movement_velocity = np.array(interpolate_nan_values(head_movement_velocity.tolist()))
    head_movement_velocity = butter_lowpass_filter(head_movement_velocity, cutoff=25, order=1)
    dataframe["headMovementVelocityFiltered"] = head_movement_velocity

    head_movement_acceleration = dataframe["hmdAngularAccelerationVectorize"].values
    head_movement_acceleration = replace_values_above_threshold_to_nan(head_movement_acceleration, threshold=1000)
    head_movement_acceleration = add_margin_around_nan_values(head_movement_acceleration, 500)
    head_movement_acceleration = np.array(interpolate_nan_values(head_movement_acceleration.tolist()))
    dataframe["headMovementAcceleration"] = head_movement_acceleration
    head_movement_acceleration = butter_lowpass_filter(head_movement_acceleration, cutoff=25, order=1)
    dataframe["headMovementAccelerationFiltered"] = head_movement_acceleration

    dataframe["headMovementAcceleration"] = dataframe["headMovementAcceleration"].rolling(20).mean()

    # plt.plot(dataframe["headMovementVelocityFiltered"])

    # plt.plot(dataframe["headMovementAcceleration"])
    # plt.plot(dataframe["headMovementAccelerationFiltered"])
    # plt.axhline(y=100, color="red")
    # plt.axhline(y=75, color="green")
    # plt.show()
    return dataframe


def filter_hand_movement_data(dataframe: pd.DataFrame):
    # dataframe = time_derivative(dataframe, "rightControllerPosition", "rightControllerVelocity")
    # dataframe = time_derivative(dataframe, "rightControllerVelocity", "rightControllerAcceleration")
    # dataframe = time_derivative(dataframe, "rightControllerAcceleration", "rightControllerJerk")

    dataframe.fillna(0, inplace=True)

    dataframe["rightControllerPositionVectorize"] = dataframe["rightControllerPosition"].apply(
        lambda x: np.linalg.norm(x)
    )
    dataframe["rightControllerPositionVectorizeFiltered"] = butter_lowpass_filter(dataframe["rightControllerPositionVectorize"], cutoff=20, order=3)
    dataframe = time_derivative(dataframe, "rightControllerPositionVectorizeFiltered", "rightControllerVelocity")
    dataframe = time_derivative(dataframe, "rightControllerVelocity", "rightControllerAcceleration")
    dataframe = time_derivative(dataframe, "rightControllerAcceleration", "rightControllerJerk")
    # dataframe["rightControllerVelocityVectorize"] = dataframe["rightControllerVelocity"].apply(
    #     lambda x: np.linalg.norm(x)
    # )
    # dataframe["rightControllerAccelerationVectorize"] = dataframe["rightControllerAcceleration"].apply(
    #     lambda x: np.linalg.norm(x)
    # )
    # dataframe["rightControllerJerkVectorize"] = dataframe["rightControllerJerk"].apply(
    #     lambda x: np.linalg.norm(x)
    # )
    return dataframe


for i in range(1, 4):
    df = create_clean_dataframe_hmd(4, i)
    df = filter_hand_movement_data(df)
    # position_signal = df["rightControllerPositionVectorize"]
    # rolling_mean = df["rightControllerPositionVectorize"].rolling(window=20).mean()
    # signal_unfiltered = df["rightControllerPositionVectorize"]
    # plt.plot(signal_unfiltered, alpha =0.5)
    signal = df["rightControllerJerk"]
    plt.plot(signal, label=condition_names[i - 1], alpha=0.5)
    # df["rightControllerPositionVectorizeFiltered"].plot(label=condition_names[i - 1], alpha=0.5)
    # print(f"The mean frame rate in seconds for condition {i} is: {df['deltaSeconds'].mean()}")
    # print(f"The std. dev. frame rate in seconds for condition {i} is: {df['deltaSeconds'].std()}")

plt.title("Jerk vectorized over time".title())
plt.xlabel("Timeframe")
plt.ylabel("Jerk")
plt.legend()
plt.show()
