import numpy as np
from dotenv import load_dotenv
import os
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.data_analysis.helper_functions.data_helpers import obtain_feature_data
from data_analysis.visualization.plotting_aoi import boxplots_aoi

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
# feature = "total time other object"
# feature = "total time list"
# feature = "fixations other object"
# feature = "seconds/item first 16"
# feature = "std dev. seconds/item first 16"

# feature_data = obtain_feature_data(feature, conditions)


def anova(data: dict):
    data_values = list(data.values())
    return stats.f_oneway(*data_values)


def post_hoc_test(feature_1: list, feature_2: list):
    return stats.ttest_ind(feature_1, feature_2)


def pairwise_tukey_test(data: dict):
    value_list = []
    key_list = []
    for key, values in data.items():
        value_list.extend(values)
        key_list.extend([key] * len(values))
    tukey = pairwise_tukeyhsd(value_list, key_list, alpha=0.05)
    return tukey


aoi = boxplots_aoi("cart")
# hrv = heart_rate_variability_boxplot("ecg_data_unfiltered.pickle", np.arange(1, 22), np.arange(1, 8))
# print(anova(aoi))
print(pairwise_tukey_test(aoi))


