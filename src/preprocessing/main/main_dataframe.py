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


def aoi_data(aoi_name):
    """
            The name_aoi parameter is one of the names of the aois in the pickle file.
            The options of this name are ["list", "cart", "main_shelf", "other_object", "invalid", "transition"]
        """
    aoi_dictionary = load_pickle("aoi_results.pickle")
    plot_dictionary = {}
    for condition in conditions:
        aoi_values = []
        for participant in participants:
            aoi_values.append(aoi_dictionary[condition][participant - 1][aoi_name])
        plot_dictionary[f"aoi_{aoi_name}_{condition}"] = aoi_values
    return pd.DataFrame(plot_dictionary)


def heart_rate_data():
    participants_hr = np.arange(1, 22)
    filtered_peaks = load_pickle("ecg_data_unfiltered.pickle")
    times, peaks = filtered_peaks[0], filtered_peaks[1]
    heart_rates, heart_rate_variabilities = {}, {}
    for condition in conditions:
        heart_rate, heart_rate_variability = [], []
        for participant in participants_hr:
            hr = calculate_mean_heart_rate(times[condition][participant-1], peaks[condition][participant-1])
            hrv = calculate_rmssd(peaks[condition][participant - 1])
            heart_rate.append(hr)
            heart_rate_variability.append(hrv)
        heart_rates[f"hr_{condition}"] = heart_rate
        heart_rate_variabilities[f"hrv_{condition}"] = heart_rate_variability
    hr, hrv = pd.DataFrame(heart_rates), pd.DataFrame(heart_rate_variabilities)
    hr_df = pd.concat([hr, hrv], axis=1, join="outer")
    return hr_df


def movement_data():
    head_accelerations = load_pickle("head_acceleration_mean_results.pickle")
    head_accelerations = {'head_acc_' + str(key): value for key, value in head_accelerations.items()}
    head_accelerations_df = pd.DataFrame(head_accelerations)

    hand_movement_mean_grab_time = load_pickle("box_plot_hand_movements_grab_time.pickle")
    hand_movement_mean_grab_time = {'hand_grab_time_' + str(key): value for key, value in hand_movement_mean_grab_time.items()}
    hand_movement_mean_grab_time_df = pd.DataFrame(hand_movement_mean_grab_time)

    hand_movement_rmse = load_pickle("box_plot_hand_movements_rmse.pickle")
    hand_movement_rmse = {'hand_rmse_' + str(key): value for key, value in hand_movement_rmse.items()}
    hand_movement_rmse_df = pd.DataFrame(hand_movement_rmse)

    movement_df = pd.concat([head_accelerations_df, hand_movement_mean_grab_time_df, hand_movement_rmse_df], axis=1)
    return movement_df


def nasa_tlx_data():
    nasa_tlx_dict = {}
    for condition in conditions:
        nasa_tlx_condition = []
        for participant in participants:
            nasa_tlx_condition.append(nasa_tlx_weighted(participant, condition))
        nasa_tlx_dict[f"nasa_tlx_{condition}"] = nasa_tlx_condition
    return pd.DataFrame(nasa_tlx_dict)


def performance_data():
    performance_dict = {}
    for condition in conditions:
        p = load_pickle(f"c{condition}.pickle")
        filtered_data = [x for x in p["seconds/item window"] if x is not None]
        performance_dict[f"performance_{condition}"] = filtered_data
    return pd.DataFrame(performance_dict)


def behavior_data():
    overlap_grab_list = load_pickle("percentage_list_isgrabbing.pickle")
    overlap_grab_list = {'overlap_grab_list_' + str(key): value for key, value in overlap_grab_list.items()}
    overlap_grab_list_df = pd.DataFrame(overlap_grab_list)

    ratio_frequency_list_items = load_pickle("ratio_frequency_list_items.pickle")
    ratio_frequency_list_items = {'ratio_frequency_list_items_' + str(key): value for key, value in ratio_frequency_list_items.items()}
    ratio_frequency_list_items_df = pd.DataFrame(ratio_frequency_list_items)

    ratio_time_list_items = load_pickle("ratio_time_list_items.pickle")
    ratio_time_list_items = {'ratio_time_list_items_' + str(key): value for key, value in ratio_time_list_items.items()}
    ratio_time_list_items_df = pd.DataFrame(ratio_time_list_items)

    behavior_df = pd.concat([overlap_grab_list_df, ratio_frequency_list_items_df, ratio_time_list_items_df], axis=1)
    write_pickle("main_dataframe_2.pickle", behavior_df)
    return behavior_df


def condition_order_number(participant, condition):
    filename = os.path.join(DATA_DIRECTORY, "nasa_tlx", "participants-conditions.xlsx")
    condition_orders = pd.read_excel(filename, sheet_name="order-conditions-T", index_col=0)
    n_in_order = np.where(condition_orders[f"p{participant}"] == condition)[0][0] + 1
    # print(f"participant {participant}, condition {condition}: {n_in_order}th in the order")
    return n_in_order


def participant_order(participant):
    filename = os.path.join(DATA_DIRECTORY, "nasa_tlx", "participants-conditions.xlsx")
    condition_orders = pd.read_excel(filename, sheet_name="order-conditions-T", index_col=0)
    return condition_orders[f"p{participant}"].values


def create_main_dataframe_1():
    """
    Creates a dataframe with all the mean data.
    Columns exist of the name of the measure, including a condition number.

    Measure names = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
    "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"].

    Example: The column name for the fifth condition of the heart rate is "hr_5"
    """
    aoi_names = ["cart", "list", "main_shelf", "other_object"]
    aoi_dfs = []
    for name in aoi_names:
        aoi_dfs.append(aoi_data(name))
    aoi_df = pd.concat(aoi_dfs, axis=1, join="outer")

    hr_df = heart_rate_data()
    movements_df = movement_data()
    nasa_tlx_df = nasa_tlx_data()
    performance_df = performance_data()

    main_dataframe = pd.concat([aoi_df, hr_df, movements_df, nasa_tlx_df, performance_df], axis=1, join="outer")
    write_pickle("main_dataframe.pickle", main_dataframe)
    return main_dataframe


def create_long_df(main_df):
    reshaped_data = []
    for index, row in main_df.iterrows():
        participant = index + 1
        for condition in range(1, 8):
            new_row = {
                "participant": participant,
                "condition": condition,
                "order": condition_order_number(participant, condition),
                "aoi_cart": row[f"aoi_cart_{condition}"],
                "aoi_list": row[f"aoi_list_{condition}"],
                "aoi_main_shelf": row[f"aoi_main_shelf_{condition}"],
                "aoi_other_object": row[f"aoi_other_object_{condition}"],
                "hr": row[f"hr_{condition}"],
                "hrv": row[f"hrv_{condition}"],
                "head_acc": row[f"head_acc_{condition}"],
                "head_idle": row[f"head_idle_{condition}"],
                "hand_grab_time": row[f"hand_grab_time_{condition}"],
                "hand_rmse": row[f"hand_rmse_{condition}"],
                "hand_jerk": row[f"hand_jerk_{condition}"],
                "nasa_tlx": row[f"nasa_tlx_{condition}"],
                "performance": row[f"performance_{condition}"]
            }
            reshaped_data.append(new_row)
    long_df = pd.DataFrame(reshaped_data)
    write_pickle("main_dataframe_long.pickle", long_df)
    return long_df


def create_long_df_2(main_df):
    reshaped_data = []
    for index, row in main_df.iterrows():
        participant = index + 1
        for condition in range(1, 8):
            new_row = {
                "participant": participant,
                "condition": condition,
                "order": condition_order_number(participant, condition),
                "overlap_grab_list": row[f"overlap_grab_list_{condition}"],
                "ratio_frequency_list_items": row[f"ratio_frequency_list_items_{condition}"],
                "ratio_time_list_items": row[f"ratio_time_list_items_{condition}"]
            }
            reshaped_data.append(new_row)
    long_df = pd.DataFrame(reshaped_data)
    write_pickle("main_dataframe_long_2.pickle", long_df)
    return long_df


# condition_order_number(6, 2)
# main_df = load_pickle("main_dataframe.pickle")
# main_df.at[5, "aoi_other_object_4"] = np.mean(main_df["aoi_other_object_4"])
# # print(main_df["aoi_other_object_4"])
# write_pickle("main_dataframe.pickle", main_df)
# behavior_data()
# create_long_df_2(load_pickle("main_dataframe_2.pickle"))
# long_df = create_long_df(load_pickle("main_dataframe.pickle"))
# print(long_df)



