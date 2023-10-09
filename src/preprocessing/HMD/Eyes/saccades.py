import numpy as np
import pandas as pd

from src.preprocessing.HMD.Eyes.fixations import (
    create_clean_dataframe_hmd,
    add_filter_average_to_dataframe,
    add_degrees_per_second_to_dataframe,
    add_gaze_position_to_dataframe,
    add_angle_to_dataframe,
)


#  TODO: is calculating saccades useful for the algorithm? Maybe fixations is enough

def saccades(participant_no, condition, start_index, end_index):
    #  TODO: this function is not finished at all, first plot the degrees/second and pick thresholds for the
    #   saccade velocity and time thresholds.
    max_rotational_velocity = 50
    features = {}
    clean_dataframe = create_clean_dataframe_hmd(participant_no, condition)[start_index:end_index]
    add_gaze_position_to_dataframe(clean_dataframe)
    add_filter_average_to_dataframe(
        clean_dataframe,
        "gazePosition",
        "gazePositionAverage",
        3
    )
    add_degrees_per_second_to_dataframe(clean_dataframe, "gazePositionAverage")