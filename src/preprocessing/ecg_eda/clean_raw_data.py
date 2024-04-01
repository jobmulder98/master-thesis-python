import pandas as pd
import os
from biosppy.signals import ecg
from dotenv import load_dotenv

from src.preprocessing.helper_functions.dataframe_helpers import convert_column_to_float
from src.preprocessing.helper_functions.general_helpers import text_file_name

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


def create_clean_dataframe_ecg_eda(participant_no: int) -> pd.DataFrame:
    dataframe = text_to_dataframe(participant_no)

    # Convert column values to float
    float_columns = ["[B] Heart Rate",
                     "[B] HRV Amp.",
                     "[B] HRV-LF Power (0.04-0.16 Hz)",
                     "[B] HRV-HF Power (0.16-0.4 Hz)",
                     "[B] HRV-LF / HRV-HF "]
    for column in float_columns:
        convert_column_to_float(dataframe, column)

    #  Convert to float and filter ecg signal and replace in dataframe
    if "Sensor-B:EEG" in dataframe.columns:
        convert_column_to_float(dataframe, "Sensor-B:EEG")

    # Convert to float and filter eda signal and replace in dataframe
    if "Sensor-C:SC/GSR" in dataframe.columns:
        convert_column_to_float(dataframe, "Sensor-C:SC/GSR")

    dataframe["userId"] = participant_no
    return dataframe


def filter_ecg_signal(ecg_signal):
    return ecg.ecg(ecg_signal, sampling_rate=ECG_SAMPLE_RATE, show=False)[1]
