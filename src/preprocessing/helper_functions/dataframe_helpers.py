import numpy as np
import numpy.typing as npt
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")


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


def interpolate_zeros(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe[column_name] = dataframe[column_name].replace(0, np.nan)
    dataframe[column_name] = dataframe[column_name].interpolate(method="linear")
    return dataframe


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
