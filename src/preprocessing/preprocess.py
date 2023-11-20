import numpy as np
import numpy.typing as npt
import pandas as pd
import pickle
from dotenv import load_dotenv
import os
from tqdm import tqdm

from src.preprocessing.helper_functions.data_merge_helpers import (
    merge_all_features_into_dictionary,
    merge_dictionaries,
)
from src.preprocessing.helper_functions.time_synchronize_helpers import synchronize_all_conditions

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def initialize_synchronized_times(participants, overwrite_old_pickle=False) -> dict:
    if os.path.exists(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle"):
        if not overwrite_old_pickle:
            with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
                return pickle.load(handle)

    synchronized_times = {}
    for participant in participants:
        synchronized_times[participant] = synchronize_all_conditions(participant)
    with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "wb") as handle:
        pickle.dump(synchronized_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return synchronized_times


def initialize_data_per_condition(participants: npt.NDArray,
                                  conditions: npt.NDArray,
                                  synchronized_times: dict,
                                  start_idx: int,
                                  end_idx: int,
                                  fixation_time_thresholds: dict,
                                  overwrite_old_pickle=True):
    """
    Create pickle files with the data of all participants per condition.
    Inputs numbers of participants and conditions, synchronized times for the ecg and eda
    signal, fixation settings, and the time window if required

    Only run if the pickles do not exist, skip this function otherwise.
    """
    for condition in tqdm(conditions,
                          ncols=100,
                          desc="Processing conditions",
                          unit="condition",
                          colour="white"
                          ):
        features = []
        for participant in tqdm(participants,
                                ncols=100,
                                desc="Processing participants",
                                unit="participant",
                                colour="green"
                                ):
            start_idx_ecg_eda, end_idx_ecg_eda = synchronized_times[participant][condition]
            features.append(merge_all_features_into_dictionary(
                participant,
                condition,
                start_idx_ecg_eda,
                end_idx_ecg_eda,
                start_idx,
                end_idx,
                fixation_time_thresholds,
            ))
        all_features_participant = merge_dictionaries(features)
        if overwrite_old_pickle:
            with open(f"{DATA_DIRECTORY}\pickles\c{condition}.pickle", "wb") as handle:
                pickle.dump(all_features_participant, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    participants = np.arange(1, 23)
    conditions = np.arange(7, 8)
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
    synchronized_times = initialize_synchronized_times(participants, overwrite_old_pickle=False)
    # print(synchronized_times)

    initialize_data_per_condition(participants=participants,
                                  conditions=conditions,
                                  start_idx=start_idx,
                                  end_idx=end_idx,
                                  synchronized_times=synchronized_times,
                                  fixation_time_thresholds=fixation_time_thresholds,
                                  overwrite_old_pickle=True
                                  )
