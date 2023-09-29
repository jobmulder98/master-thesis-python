import pandas as pd
import numpy as np

from src.preprocessing.HMD.clean_raw_data import create_clean_dataframe
from src.preprocessing.helper_functions.general_helpers import (
    angle_between_points,
    milliseconds_to_seconds,
)

# pandas warning setting
pd.options.mode.chained_assignment = None


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


def add_filter_average_to_dataframe(
        dataframe: pd.DataFrame,
        column_name: str,
        new_column_name: str,
        average_over_n_values: int
):
    new_column_list = []
    for i in range(len(dataframe[column_name])):
        if (i + 1) < average_over_n_values:
            new_column_list.append(np.array([0, 0, 0]))
        else:
            gaze_points_added_together = 0
            gaze_points = dataframe[column_name].iloc[(i + 1 - average_over_n_values):(i + 1)]
            for gaze_point in gaze_points:
                gaze_point = gaze_point.astype('float64')
                gaze_points_added_together += gaze_point
            gaze_points_added_together /= average_over_n_values
            new_column_list.append(gaze_points_added_together)
    dataframe[new_column_name] = new_column_list
    return


def add_angle_to_dataframe(dataframe: pd.DataFrame, gaze_position_column: str) -> None:
    list_of_angles = [0]  # first angle between points is NaN
    for i in range(len(dataframe[gaze_position_column]) - 1):
        list_of_angles.append(angle_between_points(
            dataframe[gaze_position_column].iloc[i],
            dataframe[gaze_position_column].iloc[i + 1],
            dataframe["rayOrigin"].iloc[i]
        ))
    dataframe["angle"] = list_of_angles
    return


def add_degrees_per_second_to_dataframe(dataframe: pd.DataFrame, gaze_position_column: str) -> None:
    add_angle_to_dataframe(dataframe, gaze_position_column)
    dataframe["degreesPerSecond"] = dataframe["angle"] / dataframe["deltaSeconds"]
    dataframe["degreesPerSecond"].iloc[0] = 0
    dataframe["degreesPerSecond"] = dataframe["degreesPerSecond"].apply(lambda x: min(x, 700))
    return


def count_fixations(dataframe: pd.DataFrame,
                    fixations_column_name: str,
                    max_rotational_velocity: int,
                    min_threshold_milliseconds: int,
                    max_threshold_milliseconds: int,
                    on_other_object=False) -> tuple:
    time_counter = 0
    fixation_counter = 0
    fixation_times = []
    min_threshold = milliseconds_to_seconds(min_threshold_milliseconds)
    max_threshold = milliseconds_to_seconds(max_threshold_milliseconds)
    for i in range(len(dataframe[fixations_column_name])):
        if dataframe[fixations_column_name].iloc[i] < max_rotational_velocity:
            if on_other_object:
                if dataframe["focusObjectTag"].iloc[i] == "notAssigned":
                    time_counter += dataframe["deltaSeconds"].iloc[i]
            else:
                time_counter += dataframe["deltaSeconds"].iloc[i]
        else:
            if min_threshold <= time_counter < max_threshold:
                fixation_counter += 1
                fixation_times.append(time_counter)
            time_counter = 0
    return fixation_counter, fixation_times


def fixation_features(participant_no, condition, start_index, end_index, plot=False):
    fixation_time_thresholds = {
        "short fixations": [100, 150, False],  # Here False corresponds to on_other_object
        "medium fixations": [150, 300, False],
        "long fixations": [300, 500, False],
        "very long fixations": [500, 2000, False],
        "all fixations": [100, 2000, False],
        "fixations other object": [100, 2000, True],
    }
    max_rotational_velocity = 50
    features = {}
    clean_dataframe = create_clean_dataframe(participant_no, condition)[start_index:end_index]
    add_gaze_position_to_dataframe(clean_dataframe)
    add_filter_average_to_dataframe(
        clean_dataframe,
        "gazePosition",
        "gazePositionAverage",
        3
    )
    add_degrees_per_second_to_dataframe(clean_dataframe, "gazePositionAverage")

    for key, value in fixation_time_thresholds.items():
        number_of_fixations, fixation_times = count_fixations(clean_dataframe,
                                                              "degreesPerSecond",
                                                              max_rotational_velocity,
                                                              value[0],
                                                              value[1],
                                                              value[2],
                                                              )
        features[key] = number_of_fixations
        if key == "all fixations":
            features["mean fixation time"] = np.mean(fixation_times) * 1000
            features["median fixation time"] = np.median(np.sort(fixation_times)) * 1000
        elif key == "fixations other object":
            features["mean fixation time other object"] = np.mean(fixation_times) * 1000
            features["median fixation time other object"] = np.median(np.sort(fixation_times)) * 1000
            features["longest fixation other object"] = np.max(fixation_times) * 1000

    if plot:
        return features  # TODO
    return features


participant_number = 103
condition = 3
start = 0
end = -1

print("Fixation features:")
for k, v in fixation_features(participant_no=participant_number,
                              condition=condition,
                              start_index=start,
                              end_index=end).items():
    print("- %s: %.2f" % (k, v))

