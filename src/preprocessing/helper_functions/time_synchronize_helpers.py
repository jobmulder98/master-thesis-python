import numpy as np
import pandas as pd
import dateutil.parser as dparser
import linecache
import os

from datetime import datetime
from dotenv import load_dotenv

from src.preprocessing.helper_functions.dataframe_helpers import (
    convert_column_to_datetime,
)
from src.preprocessing.helper_functions.general_helpers import (
    delta_time_seconds,
    text_file_name,
    csv_file_name,
)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def obtain_start_end_times_hmd(participant_number: int, condition: int) -> tuple[datetime, datetime]:
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


def synchronize_times(start_time_condition: datetime,
                      end_time_condition: datetime,
                      start_time_ecg: datetime) -> tuple[int, int]:
    difference_in_seconds = delta_time_seconds(start_time_ecg, start_time_condition)
    start_at_index = difference_in_seconds * ECG_SAMPLE_RATE
    total_time_condition = delta_time_seconds(start_time_condition, end_time_condition)
    end_at_index = start_at_index + total_time_condition * ECG_SAMPLE_RATE
    return int(start_at_index), int(end_at_index)


def synchronize_all_conditions(participant_number: int) -> dict:
    start_end_times = {}
    for condition in np.arange(1, 8):
        start_time_condition, end_time_condition = obtain_start_end_times_hmd(participant_number, condition)
        start_time_ecg = obtain_start_time_ecg(participant_number)
        index_start, index_end = synchronize_times(start_time_condition, end_time_condition, start_time_ecg)
        start_end_times[condition] = [index_start, index_end]
    return start_end_times


# print(synchronize_all_conditions(103))
