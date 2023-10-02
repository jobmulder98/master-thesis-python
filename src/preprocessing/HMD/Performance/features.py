import numpy as np
import pandas as pd

from src.preprocessing.HMD.clean_raw_data import *
from src.preprocessing.helper_functions.general_helpers import delta_time_seconds


def new_item_in_cart(dataframe: pd.DataFrame, performance_column: str, difference_column: str) -> tuple:
    dataframe[difference_column] = dataframe[performance_column].diff()

    # TODO: should we add the first item as performance? This time might not be representative (start up time).
    dataframe.loc[0, difference_column] = 1  # If the window starts from time stamp 0, this is the start point

    new_item_in_cart_indexes, new_item_in_cart_time_stamp = [], []
    for index, row in dataframe.iterrows():
        if row[difference_column] == 1:
            new_item_in_cart_indexes.append(int(index))
            new_item_in_cart_time_stamp.append(row["timeStampDatetime"])
    return new_item_in_cart_indexes, new_item_in_cart_time_stamp


def delta_time_new_item(participant_no: int, condition: int, start_index, end_index):
    dataframe = create_clean_dataframe(participant_no, condition)[start_index:end_index]
    performance_column, difference_column = "numberOfItemsInCart", "numberOfItemsInCartDifference"
    indexes, time_stamps = new_item_in_cart(dataframe, performance_column, difference_column)
    delta_times = []
    for i in range(len(time_stamps) - 1):
        delta_times.append(delta_time_seconds(time_stamps[i], time_stamps[i+1]))
    return delta_times


def performance_features(participant_no: int, condition: int, start_index: int, end_index: int) -> dict:
    delta_times = delta_time_new_item(participant_no, condition, start_index, end_index)
    time_sum = sum(map(float, delta_times))
    features = {
        "seconds/item": time_sum / len(delta_times),
        "std dev. seconds/item" : np.std(delta_times),
    }
    return features


participant_number = 103
condition = 3
start_idx = 0
end_idx = 3000

print("Performance features:")
for k, v in performance_features(participant_number, condition, start_idx, end_idx).items():
    print("- %s: %.2f" % (k, v))
