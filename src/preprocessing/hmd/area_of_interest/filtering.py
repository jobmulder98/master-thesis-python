import numpy as np
import pandas as pd

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.helper_functions.general_helpers import write_pickle


def total_time_other_object(dataframe: pd.DataFrame) -> float:
    condition = (dataframe["focusObjectTag"] == "notAssigned") | \
                (dataframe["focusObjectTag"] == "NPC") | \
                (dataframe["focusObjectTag"] == "Alarm")
    return dataframe.loc[condition, "deltaSeconds"].sum()


def total_time_not_assigned(dataframe: pd.DataFrame) -> float:
    condition = dataframe["focusObjectTag"] == "notAssigned"
    return dataframe.loc[condition, "deltaSeconds"].sum()


def total_time_npc(dataframe: pd.DataFrame) -> float:
    condition = dataframe["focusObjectTag"] == "NPC"
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
            "destination T402": "walking character behind main shelf",
            "destination T403": "walking character behind main shelf"
        }
    if dataframe["condition"][1] == 3:
        destination_to_character = {
            "destinationT102": "walking character behind main shelf",
            "destinationT103": "walking character behind main shelf",
            "destination T402": "walking character behind main shelf",
            "destination T403": "walking character behind main shelf",
            "destinationT502": "dancing character",
            "destinationT504": "pilot woman",
            "destinationT602": "pilot woman",
            "destinationT702": "dancing character",
        }
    dataframe["focusObjectName"].replace(destination_to_character, inplace=True)
    return dataframe


def replace_character_to_aoi(dataframe: pd.DataFrame) -> pd.DataFrame:
    #  TODO: categorize characters to aoi in order to plot in histogram
    pass


def create_data_pickle():
    participants = np.arange(1, 23)
    conditions = np.arange(1, 8)
    aoi_results = {}
    for condition in conditions:
        condition_results = []
        for participant in participants:
            hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
            filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["notAssigned", "NPC"], 0.1)
            filtered_hmd_dataframe = replace_destination_with_character(filtered_hmd_dataframe)

            condition_result = {"list": total_time_list(filtered_hmd_dataframe),
                                "cart": total_time_cart(filtered_hmd_dataframe),
                                "main_shelf": total_time_main_shelf(filtered_hmd_dataframe),
                                "other_object": total_time_other_object(filtered_hmd_dataframe),
                                "invalid": total_time_invalid(filtered_hmd_dataframe),
                                "transition": total_time_transition(filtered_hmd_dataframe),
                                }
            condition_results.append(condition_result)
        aoi_results[condition] = condition_results
    write_pickle("aoi_results.pickle", aoi_results)
