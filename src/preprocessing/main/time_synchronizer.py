import pickle
from dotenv import load_dotenv
import os

from src.preprocessing.helper_functions.time_synchronize_helpers import synchronize_all_conditions

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def initialize_synchronized_times(participants, overwrite_old_pickle=False) -> dict:
    # Check if the synchronized times pickle file already exists
    if os.path.exists(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle"):
        # If overwrite_old_pickle is False and the file exists, load and return the data from the pickle file
        if not overwrite_old_pickle:
            with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "rb") as handle:
                return pickle.load(handle)

    synchronized_times = {}  # Initialize an empty dictionary to store synchronized times

    # Iterate over each participant
    for participant in participants:
        # Synchronize all conditions for the participant and store the result in the dictionary
        synchronized_times[participant] = synchronize_all_conditions(participant)

    # Write the synchronized times dictionary to a pickle file
    with open(f"{DATA_DIRECTORY}\pickles\synchronized_times.pickle", "wb") as handle:
        pickle.dump(synchronized_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return synchronized_times  # Return the synchronized times dictionary

