import numpy as np
import pandas as pd

from src.preprocessing.hmd.eyes.area_of_interest import (
    total_time_cart,
    total_time_list,
    total_time_main_shelf,
    total_time_other_object,
)
from src.preprocessing.hmd.eyes.fixations import (
    add_degrees_per_second_to_dataframe,
    add_gaze_position_to_dataframe,
    add_filter_average_to_dataframe,
    count_fixations,
)
from src.preprocessing.hmd.eyes.convergence_distance import mean_convergence_distance


def area_of_interest_features(dataframe: pd.DataFrame):
    features = {"total time other object": total_time_other_object(dataframe),
                "total time main shelf": total_time_main_shelf(dataframe),
                "total time list": total_time_list(dataframe),
                "total time cart": total_time_cart(dataframe)}
    return features


def fixation_features(dataframe: pd.DataFrame, fixation_time_thresholds, plot=False):
    max_rotational_velocity = 50
    features = {}
    add_gaze_position_to_dataframe(dataframe)
    add_filter_average_to_dataframe(
        dataframe,
        "gazePosition",
        "gazePositionAverage",
        3
    )
    add_degrees_per_second_to_dataframe(dataframe, "gazePositionAverage")

    for key, value in fixation_time_thresholds.items():
        number_of_fixations, fixation_times = count_fixations(dataframe,
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
            if fixation_times:
                features["mean fixation time other object"] = np.mean(fixation_times) * 1000
                features["median fixation time other object"] = np.median(np.sort(fixation_times)) * 1000
                features["longest fixation other object"] = np.max(fixation_times) * 1000
            else:
                features["mean fixation time other object"] = 0
                features["median fixation time other object"] = 0
                features["longest fixation other object"] = 0

    features["mean convergence distance"] = mean_convergence_distance(dataframe)

    if plot:
        return features  # TODO
    return features


# participant_number = 7
# condition = 3
# start = 0
# end = -1
#
# fixation_time_thresholds = {
#         "short fixations": [100, 150, False],  # Here False corresponds to on_other_object
#         "medium fixations": [150, 300, False],
#         "long fixations": [300, 500, False],
#         "very long fixations": [500, 2000, False],
#         "all fixations": [100, 2000, False],
#         "fixations other object": [100, 2000, True],
#     }
#
# print("Fixation features:")
# df = create_clean_dataframe_hmd(participant_number, condition)
# for k, v in fixation_features(dataframe=df,
#                               fixation_time_thresholds=fixation_time_thresholds).items():
#     print("- %s: %.2f" % (k, v))
#
# print("\nAOI features:")
# for k, v in area_of_interest_features(dataframe=df).items():
#     print("- %s: %.2f" % (k, v))