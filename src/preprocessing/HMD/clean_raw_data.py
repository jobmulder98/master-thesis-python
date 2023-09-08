import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

from src.preprocessing.helper_functions.dataframe_helpers import (
    convert_column_to_array,
    convert_column_to_datetime,
    convert_column_to_float_and_replace_commas,
    convert_column_to_boolean,
    convert_column_to_integer,
    interpolate_zeros,
    interpolate_zero_arrays,
)
from src.preprocessing.helper_functions.general_helpers import delta_time_seconds

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
data_file = f"{DATA_DIRECTORY}\P0\datafile_C1.csv"


def create_clean_dataframe():
    clean_dataframe = create_dataframe(data_file)

    coordinate_column_names = ["rayOrigin", "rayDirection", "eyesDirection", "HMDposition", "HMDrotation",
                               "LeftControllerPosition", "LeftControllerRotation", "RightControllerPosition",
                               "RightControllerRotation"]
    for column in coordinate_column_names:
        convert_column_to_array(clean_dataframe, column)

    boolean_column_names = ["isLeftEyeBlinking", "isRightEyeBlinking", "isGrabbing"]
    for column in boolean_column_names:
        convert_column_to_boolean(clean_dataframe, column)

    integer_column_names = ["userID", "condition", "numberOfItemsInCart"]
    for column in integer_column_names:
        convert_column_to_integer(clean_dataframe, column)

    convert_column_to_float_and_replace_commas(clean_dataframe, "convergenceDistance")
    convert_column_to_datetime(clean_dataframe, "timeStampDatetime")

    add_delta_time_to_dataframe(clean_dataframe)
    add_cumulative_time_to_dataframe(clean_dataframe)

    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayOrigin")
    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayDirection")
    clean_dataframe = interpolate_zeros(clean_dataframe, "convergenceDistance")
    return clean_dataframe


def create_dataframe(raw_data_file: str) -> pd.DataFrame:
    dataframe = pd.read_csv(raw_data_file, delimiter=";", header=0, keep_default_na=True, index_col="frame")
    dataframe = dataframe.dropna(axis=1, how="all")
    return dataframe


def add_delta_time_to_dataframe(dataframe: pd.DataFrame) -> None:
    dataframe["deltaSeconds"] = 0
    for i in range(len(dataframe["timeStampDatetime"]) - 1):
        dataframe["deltaSeconds"].iloc[i + 1] = delta_time_seconds(
            dataframe["timeStampDatetime"].iloc[i],
            dataframe["timeStampDatetime"].iloc[i + 1]
        )
    return


def add_cumulative_time_to_dataframe(dataframe: pd.DataFrame) -> None:
    cumulative_time = 0
    cumulative_time_list = [0]
    for i in range(len(dataframe["deltaSeconds"])-1):
        cumulative_time += dataframe["deltaSeconds"].iloc[i]
        cumulative_time_list.append(cumulative_time)
    dataframe["timeCumulative"] = cumulative_time_list
    return


# dataset = create_clean_dataframe()
# print(dataset["rayOrigin"].iloc[63:75])
