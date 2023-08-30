import numpy as np
import math
import matplotlib.pyplot as plt
from clean_raw_dataset import create_clean_dataframe


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
    for i in range(len(dataframe["eyesDirection"])-1):
        dataframe["angleVelocity"].iloc[i+1] = angle_between(
            dataframe["eyesDirection"].iloc[i],
            dataframe["eyesDirection"].iloc[i+1]
        )
    return dataframe


def add_degrees_per_second_to_dataframe(dataframe):
    dataframe = add_delta_seconds_to_dataframe(dataframe)
    dataframe = add_angle_to_dataframe(dataframe)
    dataframe.loc[:, "degreesPerSecond"] = dataframe["angleVelocity"] / dataframe["deltaSeconds"]
    dataframe.loc[:, "degreesPerSecond"].iloc[0] = 0
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


clean_dataframe = create_clean_dataframe()
dataframe_with_added_columns = add_degrees_per_second_to_dataframe(clean_dataframe)
plt.plot(dataframe_with_added_columns["degreesPerSecond"])
plt.show()
