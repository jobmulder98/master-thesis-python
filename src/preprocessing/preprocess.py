import numpy as np
import pandas as pd


from src.preprocessing.ECG_EDA.ECG.features import ecg_features
from src.preprocessing.ECG_EDA.EDA.features import eda_features
from src.preprocessing.HMD.Eyes.features import fixation_features, area_of_interest_features
from src.preprocessing.HMD.Performance.features import performance_features


def merge_feature_dictionaries(participant_no: int, condition: int, start_index: int, end_index: int) -> dict:
    ecg = ecg_features(participant_no, start_index, end_index)
    eda = eda_features(participant_no, start_index, end_index)
    fixations = fixation_features(participant_no, condition, start_index, end_index, fixation_time_thresholds)
    areas_of_interest = area_of_interest_features(participant_no, condition, start_index, end_index)
    performance = performance_features(participant_no, condition, start_index, end_index)
    return {**ecg, **eda, **fixations, **areas_of_interest, **performance}


participant_number = 103
condition = 3
start_idx = 0
end_idx = -1
fixation_time_thresholds = {
        "short fixations": [100, 150, False],  # Here False corresponds to on_other_object
        "medium fixations": [150, 300, False],
        "long fixations": [300, 500, False],
        "very long fixations": [500, 2000, False],
        "all fixations": [100, 2000, False],
        "fixations other object": [100, 2000, True],
    }


print("Features:")
for k, v in merge_feature_dictionaries(participant_number, condition, start_idx, end_idx).items():
    print("- %s: %s" % (k, v))
