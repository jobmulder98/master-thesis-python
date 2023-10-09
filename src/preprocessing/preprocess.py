import numpy as np
import pandas as pd

from src.preprocessing.helper_functions.data_merge_helpers import merge_all_features_into_dictionary
from src.preprocessing.helper_functions.time_synchronize_helpers import synchronize_all_conditions


participant_number = 103
condition = 3
start_idx = 0
end_idx = -1
synchronized_times = synchronize_all_conditions(participant_number)
start_idx_ecg_eda, end_idx_ecg_eda = synchronized_times[condition]
fixation_time_thresholds = {
        "short fixations": [100, 150, False],  # Here False corresponds to on_other_object
        "medium fixations": [150, 300, False],
        "long fixations": [300, 500, False],
        "very long fixations": [500, 2000, False],
        "all fixations": [100, 2000, False],
        "fixations other object": [100, 2000, True],
    }

print("Features:")
for k, v in merge_all_features_into_dictionary(
        participant_number,
        condition,
        start_idx,  # NOTE THAT THIS SHOULD BE CHANGED TO start_idx_ecg_eda
        end_idx,    # NOTE THAT THIS SHOULD BE CHANGED TO end_idx_ecg_eda
        start_idx,
        end_idx,
        fixation_time_thresholds,
).items():
    print("- %s: %s" % (k, v))
