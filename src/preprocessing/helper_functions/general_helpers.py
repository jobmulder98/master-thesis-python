import numpy as np
import numpy.typing as npt
import math
import datetime
import os
import pickle
from scipy.signal import butter, filtfilt, lfilter
from dotenv import load_dotenv

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def angle_between_points(point1: npt.NDArray, point2: npt.NDArray, reference_point: npt.NDArray) -> float:
    vector1 = point1 - reference_point
    vector2 = point2 - reference_point
    dot_product = np.dot(vector1, vector2)
    magnitudes_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_angle = dot_product / magnitudes_product
    angle_in_radians = np.arccos(cos_angle)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def angle_between_vectors(v1: npt.NDArray, v2: npt.NDArray) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def butter_lowpass_filter(raw_data) -> npt.NDArray:
    fs = 1024
    cutoff = 2
    nyq = 0.5 * fs
    order = 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, raw_data)
    return filtered_data


def csv_file_name(participant_number, condition):
    return f"{DATA_DIRECTORY}\p{participant_number}\datafile_C{condition}.csv"


def delta_time_seconds(time1: datetime.datetime, time2: datetime.datetime) -> float:
    dt = time2 - time1
    dt_seconds = dt.total_seconds()
    return dt_seconds


def format_time(seconds, _):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{int(seconds):02d}"


def interpolate_nan_values(data: list) -> list:
    data_array = np.array(data, dtype=float)
    nan_indices = np.isnan(data_array)
    indices = np.arange(len(data_array))
    data_array[nan_indices] = np.interp(indices[nan_indices], indices[~nan_indices], data_array[~nan_indices])
    interpolated_data = data_array.tolist()
    return interpolated_data


def is_zero_array(array):
    return np.array_equal(array, np.array([0, 0, 0]))


def load_pickle(pickle_name):
    try:
        with open(f"{DATA_DIRECTORY}\pickles\{pickle_name}", "rb") as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        return "file does not exists"


def milliseconds_to_seconds(milliseconds):
    return milliseconds/1000


def perpendicular_distance_3d(point, start, end):
    a = start - end
    b = point - end
    distance = np.linalg.norm(np.cross(a, b)) / np.linalg.norm(a)
    return distance


def text_file_name(participant_number):
    return f"{DATA_DIRECTORY}\p{participant_number}\p{participant_number}.txt"


def unit_vector(vector: npt.NDArray) -> npt.NDArray:
    return vector / np.linalg.norm(vector)


def write_pickle(pickle_name, data):
    with open(f"{DATA_DIRECTORY}\pickles\{pickle_name}", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
