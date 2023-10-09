from collections import defaultdict
from typing import List

import numpy as np

from src.preprocessing.ECG_EDA.ECG.features import ecg_features
from src.preprocessing.ECG_EDA.EDA.features import eda_features
from src.preprocessing.HMD.Eyes.features import area_of_interest_features, fixation_features
from src.preprocessing.HMD.Movements.head_movements import head_movement_features
from src.preprocessing.HMD.Movements.hand_movements import hand_movement_features
from src.preprocessing.HMD.Performance.features import performance_features

from src.preprocessing.HMD.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.ECG_EDA.clean_raw_data import create_clean_dataframe_ecg_eda


def merge_all_features_into_dictionary(
        participant_no: int,
        condition: int,
        start_index_ecg_eda: int,
        end_index_ecg_eda: int,
        start_index: int,
        end_index: int,
        fixation_time_thresholds: dict,
) -> dict:

    ecg_eda_dataframe = create_clean_dataframe_ecg_eda(participant_no)[start_index_ecg_eda:end_index_ecg_eda]
    ecg = ecg_features(ecg_eda_dataframe, start_index_ecg_eda, end_index_ecg_eda)
    eda = eda_features(ecg_eda_dataframe, start_index_ecg_eda, end_index_ecg_eda)

    hmd_dataframe = create_clean_dataframe_hmd(participant_no, condition)[start_index:end_index]
    areas_of_interest = area_of_interest_features(hmd_dataframe)
    fixations = fixation_features(hmd_dataframe, fixation_time_thresholds)
    hand_movement = hand_movement_features(hmd_dataframe)
    head_movement = head_movement_features(hmd_dataframe)
    performance = performance_features(hmd_dataframe)

    return {**ecg, **eda, **fixations, **areas_of_interest, **head_movement, **hand_movement, **performance}


def merge_dictionaries(dictionaries: List[dict]) -> dict:
    merged_dict = defaultdict(list)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged_dict[key].append(value)
    return dict(merged_dict)


example_dictionary_list = [dict(), dict(), dict()]


