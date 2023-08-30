import numpy as np
import math
import matplotlib.pyplot as plt
from clean_raw_dataset import create_clean_dataframe
from scipy.signal import butter, filtfilt, lfilter


def add_delta_seconds_to_dataframe(dataframe):
    dataframe["deltaSeconds"] = 0
    for i in range(len(dataframe["timeStampDatetime"])-1):
        dataframe["deltaSeconds"].iloc[i+1] = calculate_delta_time_seconds(
            dataframe["timeStampDatetime"].iloc[i],
            dataframe["timeStampDatetime"].iloc[i+1]
        )
    return dataframe


def add_angle_to_dataframe(dataframe):
    dataframe["angleVelocity"] = 0
    for i in range(len(dataframe["rayDirection"])-1):
        dataframe["angleVelocity"].iloc[i+1] = angle_between(
            dataframe["rayDirection"].iloc[i],
            dataframe["rayDirection"].iloc[i+1]
        )
    # print(dataframe["angleVelocity"])
    return dataframe


def add_degrees_per_second_to_dataframe(dataframe):
    dataframe = add_delta_seconds_to_dataframe(dataframe)
    dataframe = add_angle_to_dataframe(dataframe)
    dataframe["degreesPerSecond"] = dataframe["angleVelocity"] / dataframe["deltaSeconds"]
    dataframe["degreesPerSecond"].iloc[0] = 0
    return dataframe


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def calculate_delta_time_seconds(time1, time2):
    delta_time = time2 - time1
    delta_time_seconds = delta_time.total_seconds()
    return delta_time_seconds


def butter_lowpass_filter(data):
    fs = 50
    cutoff = 10
    nyq = 0.5 * fs
    order = 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


clean_dataframe = create_clean_dataframe()
dataframe_with_added_columns = add_degrees_per_second_to_dataframe(clean_dataframe)
# filtered_degrees_per_second = butter_lowpass_filter(dataframe_with_added_columns["degreesPerSecond"])
# print(filtered_degrees_per_second)
#
# # quarter_of_the_data = int(len(filtered_degrees_per_second) / 4)
# print(dataframe_with_added_columns["degreesPerSecond"])
# # print(filtered_degrees_per_second)
plt.plot(dataframe_with_added_columns["degreesPerSecond"]) #[:quarter_of_the_data])
# # plt.plot(filtered_degrees_per_second)
plt.axhline(100, color="red")
plt.show()
