from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import os

from preprocessing.ecg_eda.ecg.filtering import calculate_mean_heart_rate, calculate_rmssd
from preprocessing.nasa_tlx.features import nasa_tlx_weighted
from src.data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.movements.filtering_movements import filter_head_movement_data

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]
measures1 = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
                 "hand_grab_time", "hand_rmse", "nasa_tlx", "performance", "hand_jerk", "head_idle"]
measures2 = ["overlap_grab_list", "ratio_frequency_list_items", "ratio_time_list_items"]


""""
DELETE THIS FILE AFTER FINISHING PROJECT
CODE IS TEMPORARILY HARDCODED IN ORDER TO UPDATE MAIN DATAFRAME
"""


def edit_main_dataframe_1(main_dataframe):
    main_dataframe = main_dataframe[main_dataframe.columns.drop(list(main_dataframe.filter(regex="jerk")))]
    main_dataframe = main_dataframe[main_dataframe.columns.drop(list(main_dataframe.filter(regex="idle")))]

    hand_movement_jerk = load_pickle("box_plot_jerk.pickle")
    hand_movement_jerk = {'hand_jerk_' + str(key): value for key, value in
                                    hand_movement_jerk.items()}
    hand_movement_jerk_df = pd.DataFrame(hand_movement_jerk)

    head_movement_idle = load_pickle("box_plot_idle_time_100.pickle")
    head_movement_idle = {'head_idle_' + str(key): value for key, value in
                          head_movement_idle.items()}
    head_movement_idle_df = pd.DataFrame(head_movement_idle)

    main_dataframe = pd.concat([main_dataframe, hand_movement_jerk_df, head_movement_idle_df], axis=1)
    # main_dataframe = main_dataframe[main_dataframe.columns.drop(list(main_dataframe.filter(regex='TODO')))]
    # write_pickle("main_dataframe.pickle", main_dataframe)
    return main_dataframe


def add_products_per_minute(main_dataframe):
    main_dataframe["ratio_time_list_items"] = 60 / main_dataframe["ratio_time_list_items"]
    print(main_dataframe["ratio_time_list_items"])


def add_data_to_main_dataframe(pickle_name: str, measure_name: str, write_to_pickle=False, write_to_csv=False):
    measure_data = load_pickle(pickle_name)
    print(measure_data)
    df = load_pickle("main_dataframe_long.pickle")

    df[measure_name] = 0

    for key, value in measure_data.items():
        for idx, participant_value in enumerate(value):
            condition = (df["condition"] == key) & (df["participant"] == idx+1)
            df.loc[condition, measure_name] = participant_value

    if write_to_pickle:
        write_pickle("main_dataframe_long.pickle", df)

    long_df = load_pickle("main_dataframe_long.pickle")

    if write_to_csv:
        long_df.to_csv("dataframe_1_R.csv")
    return


if __name__ == "__main__":
    add_data_to_main_dataframe(
        "box_plot_hand_smoothness.pickle",
        "hand_smoothness",
        write_to_pickle=True,
        write_to_csv=True
    )
    pass

