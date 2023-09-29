import numpy as np
import pandas as pd
import dateutil.parser as dparser
import matplotlib.pyplot as plt
import linecache
import os
from biosppy.signals import ecg
from typing import Union

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


def replace_string_in_textfile(raw_data_file, search_text, replace_text):
    with open(raw_data_file, "r") as file:
        data = file.read()
        data = data.replace(search_text, replace_text)
    with open(raw_data_file, "w") as file:
        file.write(data)
    return


def text_to_dataframe(participant_number: int, delimiter=',', skip_rows=11) -> pd.DataFrame:
    raw_data_file = text_file_name(participant_number)
    replace_string_in_textfile(raw_data_file, "(0,04-0,16 Hz)", "(0.04-0.16 Hz)")
    replace_string_in_textfile(raw_data_file, "(0,16-0,4 Hz)", "(0.16-0.4 Hz)")
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


def create_clean_dataframe(participant_no: int) -> pd.DataFrame:
    dataframe = text_to_dataframe(participant_no)

    # Convert column values to float
    float_columns = ["[B] Heart Rate",
                     "[B] HRV Amp.",
                     "[B] HRV-LF Power (0.04-0.16 Hz)",
                     "[B] HRV-HF Power (0.16-0.4 Hz)",
                     "[B] HRV-LF / HRV-HF "]
    for column in float_columns:
        convert_column_to_float(dataframe, column)

    #  Convert to float and filter ECG signal and replace in dataframe
    if "Sensor-B:EEG" in dataframe.columns:
        convert_column_to_float(dataframe, "Sensor-B:EEG")
        raw_ecg_signal = dataframe["Sensor-B:EEG"]
        filtered_ecg_signal = filter_ecg_signal(raw_ecg_signal)
        dataframe["Sensor-B:EEG-Filtered"] = filtered_ecg_signal

    # Convert to float and filter EDA signal and replace in dataframe
    if "Sensor-C:SC/GSR" in dataframe.columns:
        convert_column_to_float(dataframe, "Sensor-C:SC/GSR")
    return dataframe


def filter_ecg_signal(ecg_signal):
    return ecg.ecg(ecg_signal, sampling_rate=ECG_SAMPLE_RATE, show=False)[1]


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


df = create_clean_dataframe(101)
print(df.head(5))
# plt.plot(df["Sensor-B:EEG"].iloc[0:10200])
plt.show()
