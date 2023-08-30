import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")


def create_clean_dataframe():
    clean_dataframe = create_dataframe()

    coordinate_column_names = ["rayOrigin", "rayDirection", "eyesDirection", "HMDposition", "HMDrotation",
                               "LeftControllerPosition", "LeftControllerRotation", "RightControllerPosition",
                               "RightControllerRotation"]
    for column in coordinate_column_names:
        clean_dataframe = convert_column_to_list(clean_dataframe, column)

    boolean_column_names = ["isLeftEyeBlinking", "isRightEyeBlinking", "isGrabbing"]
    for column in boolean_column_names:
        clean_dataframe = convert_column_to_boolean(clean_dataframe, column)

    integer_column_names = ["userID", "condition", "numberOfItemsInCart"]
    for column in integer_column_names:
        clean_dataframe = convert_column_to_integer(clean_dataframe, column)

    clean_dataframe = convert_column_to_float(clean_dataframe, "convergenceDistance")
    clean_dataframe = convert_column_to_datetime(clean_dataframe, "timeStampDatetime")
    return clean_dataframe


def create_dataframe():
    raw_data_file = f"{DATA_DIRECTORY}\P0\datafile_C1.csv"
    dataframe = pd.read_csv(raw_data_file, delimiter=";", header=0, keep_default_na=True, index_col="frame")
    dataframe = dataframe.dropna(axis=1, how="all")
    return dataframe


def convert_column_to_list(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].apply(
        lambda x: [float(coordinate) for coordinate in x.strip('()').split(',')])
    return dataframe


def convert_column_to_boolean(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].astype(bool)
    return dataframe


def convert_column_to_integer(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].astype(int)
    return dataframe


def convert_column_to_float(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].str.replace(",", ".").astype(float)
    return dataframe


def convert_column_to_datetime(dataframe, column_name):
    dataframe[column_name] = pd.to_datetime(dataframe[column_name])
    return dataframe
