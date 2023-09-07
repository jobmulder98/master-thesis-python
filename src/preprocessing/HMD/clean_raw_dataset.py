import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

from ..helper_functions import *

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")


def create_clean_dataframe():
    clean_dataframe = create_dataframe(f"{DATA_DIRECTORY}\P0\datafile_C1.csv")

    # The first lines are startup lines, and have very large time steps and differences
    # clean_dataframe = clean_dataframe.iloc[40:].reset_index(drop=True)

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

    convert_column_to_float(clean_dataframe, "convergenceDistance")
    convert_column_to_datetime(clean_dataframe, "timeStampDatetime")

    add_delta_time_to_dataframe(clean_dataframe)
    add_cumulative_time_to_dataframe(clean_dataframe)

    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayOrigin")
    clean_dataframe = interpolate_zero_arrays(clean_dataframe, "rayDirection")
    clean_dataframe = interpolate_zeros(clean_dataframe, "convergenceDistance")
    return clean_dataframe


def create_dataframe(raw_data_file) -> pd.DataFrame:
    dataframe = pd.read_csv(raw_data_file, delimiter=";", header=0, keep_default_na=True, index_col="frame")
    dataframe = dataframe.dropna(axis=1, how="all")
    return dataframe


def convert_column_to_array(dataframe: pd.DataFrame, column_name: str) -> None:
    dataframe[column_name] = dataframe[column_name].apply(
        lambda x: np.array([float(coordinate) for coordinate in x.strip('()').split(',')]))
    return


def convert_column_to_boolean(dataframe: pd.DataFrame, column_name: str) -> None:
    dataframe[column_name] = dataframe[column_name].astype(bool)
    return


def convert_column_to_integer(dataframe: pd.DataFrame, column_name: str) -> None:
    dataframe[column_name] = dataframe[column_name].astype(int)
    return


def convert_column_to_float(dataframe: pd.DataFrame, column_name: str) -> None:
    dataframe[column_name] = dataframe[column_name].str.replace(",", ".").astype(float)
    return


def convert_column_to_datetime(dataframe: pd.DataFrame, column_name: str) -> None:
    dataframe[column_name] = pd.to_datetime(dataframe[column_name])
    return


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


def interpolate_zero_arrays(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    def is_zero_array(arr):
        return np.array_equal(arr, np.array([0, 0, 0]))

    dataframe = dataframe.copy()
    mask = dataframe[column_name].apply(is_zero_array)

    for i in range(3):
        col_name = f"{column_name}_{i}"
        dataframe[col_name] = dataframe[column_name].apply(lambda arr: arr[i])
        dataframe.loc[mask, col_name] = np.nan
        dataframe[col_name] = dataframe[col_name].interpolate(method='linear')

    dataframe.drop(columns=[column_name], inplace=True)
    dataframe[column_name] = dataframe.apply(lambda row: np.array([row[f"{column_name}_{i}"] for i in range(3)]), axis=1)
    dataframe.drop(columns=[f"{column_name}_{i}" for i in range(3)], inplace=True)
    return dataframe


def interpolate_zeros(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe[column_name] = dataframe[column_name].replace(0, np.nan)
    dataframe[column_name] = dataframe[column_name].interpolate(method="linear")
    return dataframe


# dataset = create_clean_dataframe()
# print(dataset["rayOrigin"].iloc[63:75])
