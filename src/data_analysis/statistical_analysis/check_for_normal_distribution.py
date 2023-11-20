import numpy as np
from dotenv import load_dotenv
import os
import scipy.stats as stats

from src.data_analysis.helper_functions.data_helpers import obtain_feature_data

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)

# feature = "nasa-tlx unweighted"
# feature = "nasa-tlx weighted"
# feature = "Mean HR (beats/min)"
# feature = "STD HR (beats/min)"
# feature = "all fixations"
# feature = "mean fixation time"
feature = "total time other object"
# feature = "fixations other object"
# feature = "seconds/item first 16"
# feature = "std dev. seconds/item first 16"


def is_normally_distributed(feature_name, condition):
    feature_data = obtain_feature_data(feature_name, condition)
    statistic, p_value = stats.shapiro(feature_data)
    return p_value <= 0.05


for condition in conditions:
    print(is_normally_distributed(feature, condition))
