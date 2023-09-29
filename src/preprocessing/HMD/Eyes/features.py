import numpy as np

from src.preprocessing.HMD.Eyes.area_of_interest import (
    total_time_cart,
    total_time_list,
    total_time_main_shelf,
    total_time_other_object,
)
from src.preprocessing.HMD.Eyes.fixations import (
    add_degrees_per_second_to_dataframe,
    add_gaze_position_to_dataframe,
    add_filter_average_to_dataframe,
    count_fixations,
    create_clean_dataframe,
)


def area_of_interest_features(participant_no: int, condition: int):
    features = {"total time other object": total_time_other_object(participant_no, condition),
                "total time main shelf": total_time_main_shelf(participant_no, condition),
                "total time list": total_time_list(participant_no, condition),
                "total time cart": total_time_cart(participant_no, condition)}
    return features


def fixation_features(participant_no, condition, start_index, end_index, fixation_time_thresholds, plot=False):
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

fixation_time_thresholds = {
        "short fixations": [100, 150, False],  # Here False corresponds to on_other_object
        "medium fixations": [150, 300, False],
        "long fixations": [300, 500, False],
        "very long fixations": [500, 2000, False],
        "all fixations": [100, 2000, False],
        "fixations other object": [100, 2000, True],
    }

print("Fixation features:")
for k, v in fixation_features(participant_no=participant_number,
                              condition=condition,
                              start_index=start,
                              end_index=end,
                              fixation_time_thresholds=fixation_time_thresholds).items():
    print("- %s: %.2f" % (k, v))

print("\nAOI features:")
for k, v in area_of_interest_features(participant_no=participant_number,
                                      condition=condition,
                                      ).items():
    print("- %s: %.2f" % (k, v))