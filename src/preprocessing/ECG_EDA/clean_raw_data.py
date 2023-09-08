import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser as dparser
import linecache
import os

from datetime import datetime
from dotenv import load_dotenv

from src.preprocessing.helper_functions.dataframe_helpers import (
    convert_column_to_datetime,
    convert_column_to_float,
)
from src.preprocessing.helper_functions.general_helpers import (
    delta_time_seconds,
    text_file_name,
    csv_file_name,
)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def text_to_dataframe(participant_number, delimiter=',', skip_rows=11):
    raw_data_file = text_file_name(participant_number)
    try:
        dataframe = pd.read_csv(raw_data_file,
                                delimiter=delimiter,
                                skiprows=skip_rows,
                                low_memory=False,
                                )
        dataframe.drop(dataframe.tail(3).index, inplace=True)
        return dataframe
    except FileNotFoundError:
        print(f"File not found: {raw_data_file}")
        return None


def create_clean_dataframe(dataframe):
    float_columns = ["[B] Heart Rate",
                     "[B] HRV Amp.",
                     "[B] HRV-LF Power (0,04-0,16 Hz)",
                     "[B] HRV-HF Power (0,16-0,4 Hz)",
                     "[B] HRV-LF / HRV-HF"]
    for column in float_columns:
        convert_column_to_float(dataframe, column)
    return


def obtain_start_end_times_hmd(participant_number, condition):
    raw_data_file = csv_file_name(participant_number, condition)
    clean_dataframe = pd.read_csv(raw_data_file, delimiter=";", header=0, index_col=4, keep_default_na=True)
    convert_column_to_datetime(clean_dataframe, "timeStampDatetime")
    start_time = clean_dataframe["timeStampDatetime"].iloc[0]
    end_time = clean_dataframe["timeStampDatetime"].iloc[-1]
    return start_time, end_time


def obtain_start_time_ecg(participant_number: int) -> datetime:
    filename = text_file_name(participant_number)
    line_with_date = linecache.getline(filename, 6)
    line_with_time = linecache.getline(filename, 7)
    combined_time = line_with_date + line_with_time
    time_datetime = dparser.parse(combined_time, fuzzy=True)
    return time_datetime


def synchronize_times(start_time_condition, end_time_condition, start_time_ecg):
    difference_in_seconds = delta_time_seconds(start_time_ecg, start_time_condition)
    start_at_index = difference_in_seconds * ECG_SAMPLE_RATE
    total_time_condition = delta_time_seconds(start_time_condition, end_time_condition)
    end_at_index = start_at_index + total_time_condition * ECG_SAMPLE_RATE
    return int(start_at_index), int(end_at_index)


def mean_hr(dataframe, start_index, end_index):
    return dataframe["[B] Heart Rate"].iloc[start_index:end_index].mean()


def mean_hrv_amplitude(dataframe, start_index, end_index):
    return dataframe["[B] HRV Amp."].iloc[start_index:end_index].mean()


participant_no = 103
for condition in np.arange(1, 8):
    df = text_to_dataframe(participant_no)
    print(type(df["[B] Heart Rate"].iloc[1]))
    # create_clean_dataframe(df)

    # start_time_condition, end_time_condition = obtain_start_end_times_hmd(participant_no, condition)
    # start_time_ecg = obtain_start_time_ecg(participant_no)
    # index_start, index_end = synchronize_times(start_time_condition, end_time_condition, start_time_ecg)
    # print(index_start)
    # print(index_end)
    # print(type(df["[B] Heart Rate"].iloc[index_start]))
    # print(f"The mean heart rate for condition {condition} is {mean_hr(df, index_start, index_end)}")


# print(df.head(5))
# print(df["[B] Heart Rate"].iloc[index_start:index_end].mean())


# st, et, tt = obtain_start_end_times_hmd(103, 1)
# print(st, et, tt)
