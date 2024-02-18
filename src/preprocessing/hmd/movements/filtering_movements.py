import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import interp1d
from numpy.typing import NDArray

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


def time_derivative_dataframe_column(dataframe: pd.DataFrame, x_columnname: str, x_dot_columnname: str, dt=0.015):
    """Compute the derivative for a column in a dataframe"""
    dataframe[x_dot_columnname] = dataframe[x_columnname].diff() / dt  #dataframe["deltaSeconds"]
    return dataframe


def time_derivative_signal(signal: NDArray, dt: float) -> NDArray:
    """Compute the derivative using numpy's gradient function"""
    derivative = np.gradient(signal, dt)
    return derivative


def interpolate_equal_frame_rate(time, signal, frame_rate=0.01):
    """This function interpolate a signal so that the frame rate becomes equal to frame_rate."""
    interpolator = interp1d(time, signal, kind='linear', fill_value='extrapolate')

    # Generate new time points with equal frame rate
    total_time = 122
    common_time = np.arange(0, total_time, frame_rate)

    # Interpolate the signal at the new time points
    interpolated_signal = interpolator(common_time)

    return common_time, interpolated_signal


def filter_head_movement_data(dataframe: pd.DataFrame):
    dataframe.fillna(0, inplace=True)

    # Vectorize the angle to obtain one value
    dataframe["hmdRotationVectorized"] = dataframe["hmdEuler"].apply(
        lambda x: np.linalg.norm(x)
    )

    # Interpolate the signal so frame rate becomes equal
    times = dataframe["timeCumulative"].values
    signal = dataframe["hmdRotationVectorized"].values
    dt = 0.01
    interpolated_times, interpolated_rotation_signal = interpolate_equal_frame_rate(times, signal, frame_rate=dt)

    # Filter the blocky signal
    interpolated_rotation_signal_filtered = butter_lowpass_filter(
        interpolated_rotation_signal, cutoff=20, order=3
    )

    # Take time derivatives to obtain velocity and acceleration
    velocity_signal = time_derivative_signal(interpolated_rotation_signal_filtered, dt)
    acceleration_signal = time_derivative_signal(velocity_signal, dt)

    return interpolated_times, acceleration_signal


def filter_hand_movement_data(dataframe: pd.DataFrame):
    dataframe.fillna(0, inplace=True)
    dataframe["rightControllerPositionVectorize"] = dataframe["rightControllerPosition"].apply(
        lambda x: np.linalg.norm(x)
    )

    # Interpolate the signal so frame rate becomes equal
    times = dataframe["timeCumulative"].values
    signal = dataframe["rightControllerPositionVectorize"].values
    dt = 0.01
    interpolated_times, interpolated_position_signal = interpolate_equal_frame_rate(times, signal, frame_rate=dt)

    # Filter the blocky signal
    interpolated_position_signal_filtered = butter_lowpass_filter(
        interpolated_position_signal, cutoff=20, order=3
    )

    # Three time derivatives to obtain jerk
    velocity_signal = time_derivative_signal(interpolated_position_signal_filtered, dt)
    acceleration_signal = time_derivative_signal(velocity_signal, dt)
    jerk_signal = time_derivative_signal(acceleration_signal, dt)

    return interpolated_times, jerk_signal


# for i in range(5, 8):
#     df = create_clean_dataframe_hmd(4, i)
#     times, jerk = filter_hand_movement_data(df)
#     # plt.plot(times, jerk, alpha=0.5, label=condition_names[i-1])
# #
# plt.title("Jerk vectorized over time".title())
# plt.xlabel("Timeframe")
# plt.ylabel("Jerk")
# plt.legend()
# plt.show()
