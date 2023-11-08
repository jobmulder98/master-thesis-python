import numpy as np
from dotenv import load_dotenv
import os

from src.data_analysis.helper_functions.data_helpers import obtain_feature_data

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
feature = "fixations other object"

print(obtain_feature_data(feature, conditions))


def anova(features: dict):
    return
