from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os

from preprocessing.ecg_eda.ecg.filtering import calculate_mean_heart_rate, calculate_rmssd
from preprocessing.nasa_tlx.features import nasa_tlx_weighted
from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]


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
    participants_hr = np.arange(1, 22)  # Array of participant numbers from 1 to 21
    filtered_peaks = load_pickle("ecg_data_unfiltered.pickle")  # Load filtered ECG data from a pickle file
    times, peaks = filtered_peaks[0], filtered_peaks[1]  # Extract times and peaks from the filtered data
    heart_rates, heart_rate_variabilities = {}, {}  # Initialize dictionaries to store heart rates and variabilities

    # Iterate over different conditions
    for condition in conditions:
        heart_rate, heart_rate_variability = [], []  # Initialize lists to store heart rates and variabilities for each condition

        # Iterate over each participant
        for participant in participants_hr:
            # Calculate mean heart rate for the participant's data in the current condition
            hr = calculate_mean_heart_rate(times[condition][participant - 1], peaks[condition][participant - 1])
            # Calculate root mean square of successive differences (RMSSD) for heart rate variability
            hrv = calculate_rmssd(peaks[condition][participant - 1])
            heart_rate.append(hr)  # Append calculated heart rate to the list
            heart_rate_variability.append(hrv)  # Append calculated heart rate variability to the list

        # Store heart rates and variabilities for the current condition in dictionaries
        heart_rates[f"hr_{condition}"] = heart_rate
        heart_rate_variabilities[f"hrv_{condition}"] = heart_rate_variability

    # Convert dictionaries to Pandas DataFrames
    hr, hrv = pd.DataFrame(heart_rates), pd.DataFrame(heart_rate_variabilities)

    # Concatenate heart rate and variability DataFrames along columns
    hr_df = pd.concat([hr, hrv], axis=1, join="outer")

    return hr_df  # Return the concatenated DataFrame containing heart rates and variabilities


def movement_data():
    # Load head accelerations data from a pickle file
    head_accelerations = load_pickle("head_acceleration_mean_results.pickle")
    # Rename keys in the dictionary to include 'head_acc_' prefix
    head_accelerations = {'head_acc_' + str(key): value for key, value in head_accelerations.items()}
    # Convert the dictionary to a Pandas DataFrame
    head_accelerations_df = pd.DataFrame(head_accelerations)

    # Load hand movement mean grab time data from a pickle file
    hand_movement_mean_grab_time = load_pickle("box_plot_hand_movements_grab_time.pickle")
    # Rename keys in the dictionary to include 'hand_grab_time_' prefix
    hand_movement_mean_grab_time = {'hand_grab_time_' + str(key): value for key, value in
                                    hand_movement_mean_grab_time.items()}
    # Convert the dictionary to a Pandas DataFrame
    hand_movement_mean_grab_time_df = pd.DataFrame(hand_movement_mean_grab_time)

    # Load hand movement root mean square error (RMSE) data from a pickle file
    hand_movement_rmse = load_pickle("box_plot_hand_movements_rmse.pickle")
    # Rename keys in the dictionary to include 'hand_rmse_' prefix
    hand_movement_rmse = {'hand_rmse_' + str(key): value for key, value in hand_movement_rmse.items()}
    # Convert the dictionary to a Pandas DataFrame
    hand_movement_rmse_df = pd.DataFrame(hand_movement_rmse)

    # Concatenate all the DataFrames along columns
    movement_df = pd.concat([head_accelerations_df, hand_movement_mean_grab_time_df, hand_movement_rmse_df], axis=1)

    return movement_df  # Return the concatenated DataFrame containing movement data


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
    reshaped_data = []  # Initialize an empty list to store reshaped data

    # Iterate over each row in the main DataFrame
    for index, row in main_df.iterrows():
        participant = index + 1  # Participant number is the index plus one
        # Iterate over each condition (1 to 7)
        for condition in range(1, 8):
            # Create a new dictionary representing a row in the long format DataFrame
            new_row = {
                "participant": participant,
                "condition": condition,
                "order": condition_order_number(participant, condition),  # Calculate the order of the condition
                "aoi_cart": row[f"aoi_cart_{condition}"],  # Area of interest for the cart
                "aoi_list": row[f"aoi_list_{condition}"],  # Area of interest for the list
                "aoi_main_shelf": row[f"aoi_main_shelf_{condition}"],  # Area of interest for the main shelf
                "aoi_other_object": row[f"aoi_other_object_{condition}"],  # Area of interest for other objects
                "hr": row[f"hr_{condition}"],  # Heart rate for the condition
                "hrv": row[f"hrv_{condition}"],  # Heart rate variability for the condition
                "head_acc": row[f"head_acc_{condition}"],  # Head acceleration for the condition
                "head_idle": row[f"head_idle_{condition}"],  # Head idle time for the condition
                "hand_grab_time": row[f"hand_grab_time_{condition}"],  # Hand grab time for the condition
                "hand_rmse": row[f"hand_rmse_{condition}"],  # Hand root mean square error for the condition
                "hand_jerk": row[f"hand_jerk_{condition}"],  # Hand jerk for the condition
                "nasa_tlx": row[f"nasa_tlx_{condition}"],  # NASA Task Load Index for the condition
                "performance": row[f"performance_{condition}"]  # Performance score for the condition
            }
            reshaped_data.append(new_row)  # Append the new row to the list of reshaped data

    # Convert the list of reshaped data to a Pandas DataFrame
    long_df = pd.DataFrame(reshaped_data)

    # Write the long format DataFrame to a pickle file
    write_pickle("main_dataframe_long.pickle", long_df)

    return long_df  # Return the long format DataFrame


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


def transform_long_column_to_separate_columns(main_dataframe_long, column_name):
    separated_column = {}
    for condition in conditions:
        condition_data = []
        for participant in participants:
            c = (main_dataframe_long["participant"] == participant) & (main_dataframe_long["condition"] == condition)
            value_to_append = main_dataframe_long[column_name].loc[c].squeeze()
            condition_data.append(value_to_append)
        separated_column[f"{column_name}_{condition}"] = condition_data
    return pd.DataFrame(separated_column)


# long_df = load_pickle("main_dataframe_long.pickle")
# print(long_df)
