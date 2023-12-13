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
from src.preprocessing.helper_functions.general_helpers import moving_average
conditions = np.arange(1, 8)


def filter_head_movement_data(dataframe: pd.DataFrame):
    dataframe["hmdAngularVelocity"] = dataframe["hmdEuler"].diff() / dataframe["deltaSeconds"]
    dataframe["hmdAngularAcceleration"] = dataframe["hmdAngularVelocity"].diff() / dataframe["deltaSeconds"]

    dataframe["hmdAngularVelocity"].fillna(0, inplace=True)
    dataframe["hmdAngularAcceleration"].fillna(0, inplace=True)

    dataframe["hmdAngularVelocityVectorize"] = dataframe["hmdAngularVelocity"].apply(lambda x: np.linalg.norm(x))
    dataframe["hmdAngularAccelerationVectorize"] = dataframe["hmdAngularAcceleration"].apply(lambda x: np.linalg.norm(x))
    head_movement_acceleration = dataframe["hmdAngularAccelerationVectorize"].values

    head_movement_acceleration = replace_values_above_threshold_to_nan(head_movement_acceleration, threshold=1000)
    head_movement_acceleration = add_margin_around_nan_values(head_movement_acceleration, 500)
    head_movement_acceleration = np.array(interpolate_nan_values(head_movement_acceleration.tolist()))

    dataframe["headMovementAcceleration"] = head_movement_acceleration
    dataframe["headMovementAcceleration"] = dataframe["headMovementAcceleration"].rolling(20).mean()

    # hm_peaks = head_movement_peaks(head_movement_acceleration, 250)
    # hm_peaks, _ = scipy.signal.find_peaks(head_movement_acceleration, height=200, distance=30)
    return dataframe


df = create_clean_dataframe_hmd(10, 5)
filter_head_movement_data(df)
