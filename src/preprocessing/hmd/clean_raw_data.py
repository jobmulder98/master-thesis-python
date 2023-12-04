import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from src.preprocessing.helper_functions.dataframe_helpers import (
    convert_column_to_array,
    convert_column_to_boolean,
    convert_column_to_datetime,
    convert_column_to_float_and_replace_commas,
    convert_column_to_integer,
    convert_quaternion_column_to_euler,
    interpolate_zeros,
    interpolate_zero_arrays,
)

from src.preprocessing.helper_functions.general_helpers import delta_time_seconds, is_zero_array

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")


def create_dataframe(participant_number: int, condition: int) -> pd.DataFrame:
    data_file = DATA_DIRECTORY + "\p" + str(participant_number) + "\datafile_C" + str(condition) + ".csv"
    dataframe = pd.read_csv(data_file,
                            delimiter=";",
                            encoding="utf-8",
                            header=0,
                            )
    dataframe = dataframe.dropna(axis=1, how="all")
    dataframe.set_index(keys="frame")
    return dataframe


def add_delta_time_to_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["deltaSeconds"] = np.nan
    for i in range(len(dataframe["timeStampDatetime"]) - 1):
        dataframe.loc[i + 1, "deltaSeconds"] = delta_time_seconds(
            dataframe.loc[i, "timeStampDatetime"],
            dataframe.loc[i + 1, "timeStampDatetime"]
        )
    return dataframe


def add_cumulative_time_to_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["timeCumulative"] = dataframe["deltaSeconds"].cumsum()
    return dataframe


def filter_invalid_values_missing_data(dataframe: pd.DataFrame):
    conditions = (
            (dataframe['rayOrigin'].apply(lambda x: np.array_equal(x, np.array([0, 0, 0])))) &
            (dataframe['rayDirection'].apply(lambda x: np.array_equal(x, np.array([0, 0, 0])))
             ))
    dataframe.loc[conditions, "focusObjectTag"] = "Invalid"
    dataframe.loc[conditions, "focusObjectName"] = "Invalid"
    return dataframe


def filter_invalid_values_blinking(dataframe: pd.DataFrame):
    conditions = (
            (dataframe['isLeftEyeBlinking']) |
            (dataframe['isRightEyeBlinking'])
    )
    dataframe.loc[conditions, "focusObjectTag"] = "Invalid"
    dataframe.loc[conditions, "focusObjectName"] = "Invalid"
    return dataframe


def create_clean_dataframe_hmd(participant_number: int, condition: int) -> pd.DataFrame:
    clean_dataframe = create_dataframe(participant_number, condition)

    coordinate_column_names = ["rayOrigin", "rayDirection", "eyesDirection", "hmdPosition", "hmdRotation",
                               "leftControllerPosition", "leftControllerRotation", "rightControllerPosition",
                               "rightControllerRotation"]
    for column in coordinate_column_names:
        convert_column_to_array(clean_dataframe, column)

    boolean_column_names = ["isLeftEyeBlinking", "isRightEyeBlinking", "isGrabbing"]
    for column in boolean_column_names:
        convert_column_to_boolean(clean_dataframe, column)

    integer_column_names = ["userId", "condition", "numberOfItemsInCart"]
    for column in integer_column_names:
        convert_column_to_integer(clean_dataframe, column)

    convert_column_to_float_and_replace_commas(clean_dataframe, "convergenceDistance")
    convert_column_to_datetime(clean_dataframe, "timeStampDatetime")
    convert_quaternion_column_to_euler(clean_dataframe, "hmdRotation", "hmdEuler")

    clean_dataframe = add_delta_time_to_dataframe(clean_dataframe)
    clean_dataframe = add_cumulative_time_to_dataframe(clean_dataframe)

    clean_dataframe = filter_invalid_values_blinking(clean_dataframe)
    clean_dataframe = filter_invalid_values_missing_data(clean_dataframe)

    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayOrigin")
    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayDirection")
    clean_dataframe = interpolate_zeros(clean_dataframe, "convergenceDistance")

    return clean_dataframe

# dataset = create_clean_dataframe_hmd(103, 5)
# print(dataset.to_string()[0:10000])
