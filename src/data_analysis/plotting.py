import numpy as np
import matplotlib.pyplot as plt
import pickle
from dotenv import load_dotenv
import os

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))

with open(f"{DATA_DIRECTORY}\pickles\c1.pickle", "rb") as handle:
    dict_participant = pickle.load(handle)

print(dict_participant)


labels, data = dict_participant.keys(), dict_participant.values()
fix_other_object = dict_participant['short fixations']
plt.boxplot(fix_other_object)
plt.show()
