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
from src.preprocessing.hmd.movements.head_movements import head_movement_peaks
from src.preprocessing.helper_functions.general_helpers import moving_average, butter_lowpass_filter
conditions = np.arange(1, 8)


def filter_head_movement_data(dataframe: pd.DataFrame):
    dataframe["hmdAngularVelocity"] = dataframe["hmdEuler"].diff() / dataframe["deltaSeconds"]
    dataframe["hmdAngularAcceleration"] = dataframe["hmdAngularVelocity"].diff() / dataframe["deltaSeconds"]

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

    # plt.plot(dataframe["headMovementAcceleration"])
    # plt.plot(dataframe["headMovementAccelerationFiltered"])
    # plt.show()
    return dataframe


df = create_clean_dataframe_hmd(12, 5)
filter_head_movement_data(df)
