from collections import defaultdict
from typing import List

from src.preprocessing.ecg_eda.ecg.features import ecg_features
from src.preprocessing.ecg_eda.eda.features import eda_features
from src.preprocessing.hmd.eyes.features import area_of_interest_features, fixation_features
from src.preprocessing.hmd.movements.head_movements import head_movement_features
from src.preprocessing.hmd.movements.hand_movements import hand_movement_features
from src.preprocessing.hmd.performance.features import performance_features
from src.preprocessing.nasa_tlx.features import nasa_tlx_features

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.ecg_eda.clean_raw_data import create_clean_dataframe_ecg_eda


def merge_all_features_into_dictionary(
        participant_no: int,
        condition: int,
        start_index_ecg_eda: int,
        end_index_ecg_eda: int,
        start_index: int,
        end_index: int,
        fixation_time_thresholds: dict,
) -> dict:

    ecg_eda_dataframe = create_clean_dataframe_ecg_eda(participant_no)
    if participant_no == 12 and condition == 4:  # The ecg data in this session has not been recorded
        ecg = empty_ecg_dictionary()
    else:
        ecg, _ = ecg_features(ecg_eda_dataframe, participant_no, condition, start_index_ecg_eda, end_index_ecg_eda)
    # eda = eda_features(ecg_eda_dataframe, participant_no, condition, start_index_ecg_eda, end_index_ecg_eda)

    hmd_dataframe = create_clean_dataframe_hmd(participant_no, condition)[start_index:end_index]
    areas_of_interest = area_of_interest_features(hmd_dataframe)
    fixations = fixation_features(hmd_dataframe, fixation_time_thresholds)
    hand_movement = hand_movement_features(hmd_dataframe)
    head_movement = head_movement_features(hmd_dataframe)
    performance = performance_features(hmd_dataframe)
    nasa_tlx = nasa_tlx_features(participant_no, condition)

    return {**ecg, **fixations, **areas_of_interest, **head_movement, **hand_movement, **performance, **nasa_tlx}
    # return {**ecg, **eda, **fixations, **areas_of_interest, **head_movement, **hand_movement, **performance, **nasa_tlx}


def merge_dictionaries(dictionaries: List[dict]) -> dict:
    merged_dict = defaultdict(list)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged_dict[key].append(value)
    return dict(merged_dict)


def empty_ecg_dictionary():
    return {
        "Mean NNI (ms)": None,
        "Minimum NNI": None,
        "Maximum NNI": None,
        "Mean HR (beats/min)": None,
        "STD HR (beats/min)": None,
        "Min HR (beats/min)": None,
        "Max HR (beats/min)": None,
        "SDNN (ms)": None,
        "RMSSD (ms)": None,
        "NN50": None,
        "pNN50 (%)": None,
        "Power VLF (ms2)": None,
        "Power LF (ms2)": None,
        "Power HF (ms2)": None,
        "Power Total (ms2)": None,
        "LF/HF": None,
        "Peak VLF (Hz)": None,
        "Peak LF (Hz)": None,
        "Peak HF (Hz)": None,
    }


