from scipy.stats import zscore
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os

from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.movements.filtering_movements import filter_head_movement_data

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]


def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.std(group1) ^ 2 + (n2 - 1) * np.std(group2) ^ 2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std


def cohens_dz(group1, group2):
    mean_diff = np.mean(group1 - group2)
    std_diff = np.std(group1 - group2)
    return mean_diff / std_diff


def calculate_effect_size():
    pass
