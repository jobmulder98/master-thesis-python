import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.helper_functions.general_helpers import is_zero_array


def total_time_other_object(dataframe: pd.DataFrame) -> float:
    condition = (dataframe["focusObjectTag"] == "notAssigned") | \
                (dataframe["focusObjectTag"] == "NPC") | \
                (dataframe["focusObjectTag"] == "Alarm")
    return dataframe.loc[condition, "deltaSeconds"].sum()


def total_time_invalid(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "Invalid", "deltaSeconds"].sum()


def total_time_transition(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "Transition", "deltaSeconds"].sum()


def total_time_list(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "List", "deltaSeconds"].sum()


def total_time_cart(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "Cart", "deltaSeconds"].sum()


def total_time_main_shelf(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "MainShelf", "deltaSeconds"].sum()


def filter_location_transitions(dataframe: pd.DataFrame, tag_name, threshold_seconds=0.1) -> pd.DataFrame:
    filtered_dataframe = dataframe.copy()
    time_counter = 0
    indices_to_modify = []
    is_transition = True
    for index, row in filtered_dataframe.iterrows():
        if is_transition:
            if row["focusObjectTag"] in tag_name:
                time_counter += row["deltaSeconds"]
                indices_to_modify.append(index)
                if time_counter >= threshold_seconds:
                    is_transition = False
            else:
                if time_counter < threshold_seconds:
                    filtered_dataframe.loc[indices_to_modify, "focusObjectTag"] = "Transition"
                    filtered_dataframe.loc[indices_to_modify, "focusObjectName"] = "Transition"
                time_counter = 0
                indices_to_modify = []
                is_transition = True
        else:
            if row["focusObjectTag"] not in tag_name:
                time_counter = 0
                indices_to_modify = []
                is_transition = True
    return filtered_dataframe


def replace_destination_with_character(dataframe: pd.DataFrame):
    destination_to_character = {}
    if dataframe["condition"][1] == 2:
        destination_to_character = {
            "destinationT102": "walking character behind main shelf",
            "destinationT301": "walking character behind main shelf",
            "destinationT402": "walking character behind main shelf",
            "destinationT403": "walking character behind main shelf"
        }
    if dataframe["condition"][1] == 3:
        destination_to_character = {
            "destinationT102": "walking character behind main shelf",
            "destinationT103": "walking character behind main shelf",
            "destinationT402": "walking character behind main shelf",
            "destinationT403": "walking character behind main shelf",
            "destinationT502": "dancing character",
            "destinationT602": "pilot woman",
            "destinationT702": "dancing character",
        }
    dataframe["focusObjectName"].replace(destination_to_character, inplace=True)
    return dataframe


# def create_data_pickle():
#     participants = np.arange(1, 23)
#     conditions = np.arange(1, 8)
#     aoi_data_dictionary = {}
#     for condition in conditions:
#         condition_data = []
#         for participant in participants:
#             hmd_dataframe = create_clean_dataframe_hmd(3, 3)
#             filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["notAssigned", "NPC"], 0.1)
#             filtered_hmd_dataframe = replace_destination_with_character(filtered_hmd_dataframe)


# hmd_dataframe = create_clean_dataframe_hmd(3, 3)
# filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["notAssigned", "NPC"], 0.1)
# filtered_hmd_dataframe = replace_destination_with_character(filtered_hmd_dataframe)
# print(filtered_hmd_dataframe)