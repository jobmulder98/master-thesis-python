import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import datetime
from dotenv import load_dotenv

from src.preprocessing.HMD.clean_raw_dataset import (
    create_dataframe,
    convert_column_to_datetime
)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = os.getenv("ECG_SAMPLE_RATE")

ECG_start_time = datetime.strptime("05/09/23 17:01:40", '%m/%d/%y %H:%M:%S')


def text_to_dataframe(file_path, delimiter=',', skip_rows=11):
    try:
        dataframe = pd.read_csv(file_path, delimiter=delimiter, skiprows=skip_rows)
        return dataframe
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def obtain_start_end_times(participant_number, condition):
    raw_data_file = f"{DATA_DIRECTORY}\p{participant_number}\datafile_C{condition}.csv"
    clean_dataframe = pd.read_csv(raw_data_file, delimiter=";", header=0, index_col=4, keep_default_na=True)
    convert_column_to_datetime(clean_dataframe, "timeStampDatetime")
    start_time = clean_dataframe["timeStampDatetime"].iloc[0]
    end_time = clean_dataframe["timeStampDatetime"].iloc[-1]
    total_time = end_time - start_time
    return start_time, end_time, total_time


# df = text_to_dataframe(filename)
# plt.plot(df["[B] Heart Rate"].iloc[32*270:32*450])
# plt.show()
st, et, tt = obtain_start_end_times(103, 1)
print(st, et, tt)
