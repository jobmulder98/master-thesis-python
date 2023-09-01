import numpy as np
import numpy.typing as npt
import math
import datetime
from scipy.signal import butter, filtfilt, lfilter


def angle_between_points(point1: npt.NDArray, point2: npt.NDArray, reference_point: npt.NDArray) -> float:
    vector1 = point1 - reference_point
    vector2 = point2 - reference_point
    dot_product = np.dot(vector1, vector2)
    magnitudes_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_angle = dot_product / magnitudes_product
    angle_in_radians = np.arccos(cos_angle)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def delta_time_seconds(time1: datetime.datetime, time2: datetime.datetime) -> float:
    dt = time2 - time1
    dt_seconds = dt.total_seconds()
    return dt_seconds


def unit_vector(vector: npt.NDArray) -> npt.NDArray:
    return vector / np.linalg.norm(vector)


def angle_between_vectors(v1: npt.NDArray, v2: npt.NDArray) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def milliseconds_to_seconds(milliseconds):
    return milliseconds/1000


def butter_lowpass_filter(dataframe, column_name) -> npt.NDArray:
    data = dataframe[column_name]
    fs = 200
    cutoff = 2
    nyq = 0.5 * fs
    order = 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
