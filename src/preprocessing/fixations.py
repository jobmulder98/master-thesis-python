import datetime

import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
import pandas as pd

from clean_raw_dataset import create_clean_dataframe
from scipy.signal import butter, filtfilt, lfilter

# pandas warning setting
pd.options.mode.chained_assignment = None


def add_delta_seconds_to_dataframe(dataframe: pd.DataFrame) -> None:
    dataframe["deltaSeconds"] = 0
    for i in range(len(dataframe["timeStampDatetime"])-1):
        dataframe["deltaSeconds"].iloc[i+1] = calculate_delta_time_seconds(
            dataframe["timeStampDatetime"].iloc[i],
            dataframe["timeStampDatetime"].iloc[i+1]
        )
    return


def add_gaze_position_to_dataframe(dataframe: pd.DataFrame) -> None:
    dataframe["gazePosition"] = None
    for i in range(len(dataframe["rayOrigin"])):
        if dataframe["convergenceDistance"].iloc[i] != 0:
            dataframe["gazePosition"].iloc[i] = (
                    dataframe["rayOrigin"].iloc[i]
                    + dataframe["rayDirection"].iloc[i]
                    * dataframe["convergenceDistance"].iloc[i]
            )
        else:
            dataframe["gazePosition"].iloc[i] = np.array([0, 0, 0])
    return


def filter_average(
        dataframe: pd.DataFrame,
        column_name: str,
        new_column_name: str,
        average_over_n_values: int
):
    new_column_list = []
    for i in range(len(dataframe[column_name])):
        if i+1 < average_over_n_values:
            new_column_list.append(np.array([0, 0, 0]))
        else:
            gaze_points_added_together = 0
            gaze_points = dataframe[column_name].iloc[(i+1-average_over_n_values):(i+1)]
            for gaze_point in gaze_points:
                gaze_point = gaze_point.astype('float64')
                gaze_points_added_together += gaze_point
            gaze_points_added_together /= average_over_n_values
            new_column_list.append(gaze_points_added_together)
    dataframe[new_column_name] = new_column_list
    return


def add_angle_to_dataframe(dataframe: pd.DataFrame, gaze_position_column: str) -> None:
    list_of_angles = [0]  # first angle between points is NaN
    for i in range(len(dataframe[gaze_position_column])-1):
        list_of_angles.append(angle_between_points(
            dataframe[gaze_position_column].iloc[i],
            dataframe[gaze_position_column].iloc[i+1],
            dataframe["rayOrigin"].iloc[i]
        ))
    dataframe["angleVelocity"] = list_of_angles
    return


def add_degrees_per_second_to_dataframe(dataframe: pd.DataFrame, gaze_position_column: str) -> None:
    add_delta_seconds_to_dataframe(dataframe)
    add_angle_to_dataframe(dataframe, gaze_position_column)
    dataframe["degreesPerSecond"] = dataframe["angleVelocity"] / dataframe["deltaSeconds"]
    dataframe["degreesPerSecond"].iloc[0] = 0
    return


def unit_vector(vector: npt.NDArray) -> npt.NDArray:
    return vector / np.linalg.norm(vector)


def angle_between_vectors(v1: npt.NDArray, v2: npt.NDArray) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def angle_between_points(point1: npt.NDArray, point2: npt.NDArray, reference_point: npt.NDArray) -> float:
    vector1 = point1 - reference_point
    vector2 = point2 - reference_point
    dot_product = np.dot(vector1, vector2)
    magnitudes_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_angle = dot_product / magnitudes_product
    angle_in_radians = np.arccos(cos_angle)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def calculate_delta_time_seconds(time1: datetime.datetime, time2: datetime.datetime) -> float:
    delta_time = time2 - time1
    delta_time_seconds = delta_time.total_seconds()
    return delta_time_seconds


def butter_lowpass_filter(data):
    fs = 50
    cutoff = 10
    nyq = 0.5 * fs
    order = 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


if __name__ == "__main__":
    clean_dataframe = create_clean_dataframe()
    add_gaze_position_to_dataframe(clean_dataframe)
    # filter_average(
    #     clean_dataframe,
    #     "gazePosition",
    #     "gazePositionAverage",
    #     3
    # )
    add_degrees_per_second_to_dataframe(clean_dataframe, "gazePositionAverage")
    # print(clean_dataframe["gazePosition"])
    # print(clean_dataframe["gazePositionAverage"])
    # print(clean_dataframe["angleVelocity"].head(20))
    plt.plot(clean_dataframe["angleVelocity"])
    plt.axhline(10, color="red")
    plt.show()
