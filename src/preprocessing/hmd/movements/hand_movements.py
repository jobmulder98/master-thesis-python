import numpy as np
import pandas as pd
from numpy.linalg import norm


from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.helper_functions.general_helpers import perpendicular_distance_3d


def find_start_end_coordinates(dataframe: pd.DataFrame, time_threshold: float) -> list:
    start_end_coordinates = []
    time_counter = 0  # Time counter used in case participant accidentally picks wrong item for short time
    start_coordinate, end_coordinate = None, None
    start_coordinate_idx, end_coordinate_idx = None, None
    is_grabbing = False
    for index, row in dataframe.iterrows():
        if row["isGrabbing"]:
            if not is_grabbing:
                start_coordinate = row["rightControllerPosition"]
                start_coordinate_idx = index
            time_counter += row["deltaSeconds"]
            end_coordinate = row["rightControllerPosition"]
            end_coordinate_idx = index
            is_grabbing = True
        else:
            if time_counter >= time_threshold:
                start_end_coordinates.append({"start_coordinate": start_coordinate,
                                              "end_coordinate": end_coordinate,
                                              "start_index": start_coordinate_idx,
                                              "end_index": end_coordinate_idx,
                                              "grab_time": time_counter},
                                             )
            time_counter = 0
            is_grabbing = False
    return start_end_coordinates


def rmse_hand_trajectory(dataframe: pd.DataFrame, start_end_coordinates: list[dict]) -> float:
    #  TODO: make decision for 1. rmse of all error values, or 2. rmse of trajectories, and averaging those rmses
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


def mean_grab_time(dataframe: pd.DataFrame, start_end_coordinates) -> float:
    if not start_end_coordinates:
        return 0
    grab_time = 0
    for hand_trajectory in start_end_coordinates:
        grab_time += hand_trajectory["grab_time"]
    return grab_time / len(start_end_coordinates)


def hand_movement_features(dataframe: pd.DataFrame) -> dict:
    start_end_coordinates = find_start_end_coordinates(dataframe, time_threshold=0.75)
    return {"rmse trajectory item to cart": rmse_hand_trajectory(dataframe, start_end_coordinates),
            "mean grab time": mean_grab_time(dataframe, start_end_coordinates)}


# df = create_clean_dataframe_hmd(1, 1)
# print(hand_movement_features(df))
