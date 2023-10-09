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


def delta_time_new_item(dataframe: pd.DataFrame, performance_column, difference_column) -> list[float]:
    indexes, time_stamps = new_item_in_cart(dataframe, performance_column, difference_column)
    delta_times = []
    for i in range(len(time_stamps) - 1):
        delta_times.append(delta_time_seconds(time_stamps[i], time_stamps[i+1]))
    return delta_times


def feature_calculation(dataframe: pd.DataFrame, first_16=False) -> tuple:
    performance_column, difference_column = "numberOfItemsInCart", "numberOfItemsInCartDifference"
    delta_times = delta_time_new_item(dataframe, performance_column, difference_column)
    time_sum = sum(map(float, delta_times))
    seconds_per_item = time_sum / len(delta_times)
    seconds_per_item_std_dev = np.std(delta_times)
    items_collected = int(dataframe[performance_column].iloc[-1]) - int(dataframe[performance_column].iloc[0])
    if first_16:
        if len(delta_times) >= 16:
            time_sum = sum(map(float, delta_times[0:16]))
            seconds_per_item = time_sum / len(delta_times[0:16])
            seconds_per_item_std_dev = np.std(delta_times[0:16])
    return seconds_per_item, seconds_per_item_std_dev, items_collected


def performance_features(dataframe: pd.DataFrame) -> dict:
    seconds_per_item, seconds_per_item_std_dev, items_collected = feature_calculation(dataframe)
    seconds_per_item_16, seconds_per_item_std_dev_16, _ = feature_calculation(dataframe, first_16=True)
    features = {
        "seconds/item window": seconds_per_item,
        "std dev. seconds/item window": seconds_per_item_std_dev,
        "total items collected window": items_collected,
        "seconds/item first 16": seconds_per_item_16,
        "std dev. seconds/item first 16": seconds_per_item_std_dev_16,
    }
    return features


# participant_number = 103
# condition = 1
# start_idx = 0
# end_idx = 3000
#
# print("Performance features:")
# for k, v in performance_features(participant_number, condition, start_idx, end_idx).items():
#     print("- %s: %.2f" % (k, v))
