import pickle
from dotenv import load_dotenv
import os

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

