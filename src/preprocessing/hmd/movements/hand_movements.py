import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.hmd.movements.filtering_movements import filter_hand_movement_data
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.helper_functions.general_helpers import perpendicular_distance_3d


def find_start_end_coordinates(dataframe: pd.DataFrame, time_threshold: float = 0.75) -> list:
    start_end_coordinates = []
    time_counter = 0  # Time counter used in case participant accidentally picks wrong item for short time

    start_indices = dataframe[dataframe["isGrabbing"] & ~dataframe["isGrabbing"].shift(1, fill_value=False)].index
    end_indices = dataframe[~dataframe["isGrabbing"] & dataframe["isGrabbing"].shift(1, fill_value=False)].index

    if dataframe["isGrabbing"].iloc[-1]:
        end_indices = end_indices.append(pd.Index([len(dataframe["isGrabbing"]) - 1]))

    for start_index, end_index in zip(start_indices, end_indices):
        time_counter += dataframe.loc[start_index:end_index, "deltaSeconds"].sum()
        end_coordinate = dataframe.loc[end_index, "rightControllerPosition"]
        end_coordinate_idx = end_index

        if time_counter >= time_threshold:
            start_coordinate = dataframe.loc[start_index, "rightControllerPosition"]
            start_time = dataframe.loc[start_index, "timeCumulative"]
            end_time = dataframe.loc[end_index, "timeCumulative"]
            start_coordinate_idx = start_index
            start_end_coordinates.append({
                "start_coordinate": start_coordinate,
                "end_coordinate": end_coordinate,
                "start_index": start_coordinate_idx,
                "end_index": end_coordinate_idx,
                "start_time": start_time,
                "end_time": end_time,
                "grab_time": time_counter
            })
        time_counter = 0

    return start_end_coordinates


def rmse_hand_trajectory(dataframe: pd.DataFrame, start_end_coordinates: list[dict]) -> float:
    """
    Calculates the root mean squared error compared to a staight line

    Note: old function, not used in thesis
    """
    error = []
    for hand_trajectory in start_end_coordinates:
        for i in np.arange(hand_trajectory["start_index"], hand_trajectory["end_index"]):
            start = hand_trajectory["start_coordinate"]
            end = hand_trajectory["end_coordinate"]
            point = dataframe["rightControllerPosition"].iloc[i]
            distance = perpendicular_distance_3d(point, start, end)
            error.append(distance)
    rmse_trajectories = np.sqrt(np.mean(np.square(error)))
    return rmse_trajectories


def mean_jerk(dataframe: pd.DataFrame, start_end_coordinates: list[dict]) -> float:
    jerk_all_trajectories = []
    dataframe = filter_hand_movement_data(dataframe)

    for hand_trajectory in start_end_coordinates:
        start_index = hand_trajectory["start_index"]
        end_index = hand_trajectory["end_index"]
        jerk_current_trajectory = np.abs(dataframe["rightControllerJerk"].iloc[start_index:end_index])
        jerk_all_trajectories.append(np.mean(jerk_current_trajectory))

    return np.mean(jerk_all_trajectories)


def hand_smoothness(dataframe: pd.DataFrame, start_end_coordinates: list[dict]) -> float:
    jerk_all_trajectories = []
    interpolated_times, jerk_signal = filter_hand_movement_data(dataframe)
    dt = 0.01

    for hand_trajectory in start_end_coordinates:
        start_time = hand_trajectory["start_time"]
        end_time = hand_trajectory["end_time"]
        start_index = int(start_time / dt)
        end_index = int(end_time / dt)
        jerk_current_trajectory = np.abs(jerk_signal[start_index:end_index])
        jerk_all_trajectories.append(np.mean(jerk_current_trajectory))

    return np.mean(jerk_all_trajectories)


def mean_grab_time(start_end_coordinates) -> float:
    """
        Computes the mean time an object is grabbed over an entire condition.

        Note: old function, not used in thesis
    """
    if not start_end_coordinates:
        return 0
    grab_time = 0
    for hand_trajectory in start_end_coordinates:
        grab_time += hand_trajectory["grab_time"]
    return grab_time / len(start_end_coordinates)


def hand_movement_features(dataframe: pd.DataFrame) -> dict:
    start_end_coordinates = find_start_end_coordinates(dataframe, time_threshold=0.75)
    return {"rmse trajectory item to cart": rmse_hand_trajectory(dataframe, start_end_coordinates),
            "mean grab time": mean_grab_time(start_end_coordinates)}


