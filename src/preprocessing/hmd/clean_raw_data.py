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

from src.preprocessing.helper_functions.general_helpers import delta_time_seconds

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


def add_delta_time_to_dataframe(dataframe: pd.DataFrame) -> None:
    dataframe["deltaSeconds"] = np.nan
    for i in range(len(dataframe["timeStampDatetime"]) - 1):
        dataframe.loc[i + 1, "deltaSeconds"] = delta_time_seconds(
            dataframe.loc[i, "timeStampDatetime"],
            dataframe.loc[i + 1, "timeStampDatetime"]
        )
    return


def add_cumulative_time_to_dataframe(dataframe: pd.DataFrame) -> None:
    cumulative_time = 0
    cumulative_time_list = []
    for i in range(len(dataframe["deltaSeconds"])):
        cumulative_time += dataframe["deltaSeconds"].iloc[i]
        cumulative_time_list.append(cumulative_time)
    dataframe["timeCumulative"] = cumulative_time_list
    return


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

    add_delta_time_to_dataframe(clean_dataframe)
    add_cumulative_time_to_dataframe(clean_dataframe)

    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayOrigin")
    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayDirection")
    clean_dataframe = interpolate_zeros(clean_dataframe, "convergenceDistance")

    return clean_dataframe


# dataset = create_clean_dataframe_hmd(103, 5)
# print(dataset.to_string()[0:10000])