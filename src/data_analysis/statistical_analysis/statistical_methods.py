import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg

from src.data_analysis.helper_functions.data_helpers import obtain_feature_data
from data_analysis.visualization.plotting_aoi import boxplots_aoi
from src.preprocessing.helper_functions.general_helpers import write_pickle, load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)

main_dataframe = load_pickle("main_dataframe.pickle")
main_dataframe_long = load_pickle("main_dataframe_long.pickle")

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


def linear_mixed_model(formula, dataframe, groups):
    lmm = smf.mixedlm(formula, dataframe, groups=groups, re_formula="1 + condition")
    lmm_result = lmm.fit()
    return lmm_result


def pingouin_test(dataframe, dv='nasa_tlx', within="condition", subject="participant"):
    result_mauchly = pg.sphericity(dataframe, dv=dv, within=within, subject=subject)
    return result_mauchly


if __name__ == '__main__':
    # aoi = boxplots_aoi("cart")
    # hrv = heart_rate_variability_boxplot("ecg_data_unfiltered.pickle", np.arange(1, 22), np.arange(1, 8))
    # print(anova(aoi))
    # print(pairwise_tukey_test(aoi))
    # formula_1 = "nasa_tlx ~ aoi_cart + aoi_list + aoi_main_shelf + aoi_other_object + hr + hrv + head_acc + hand_grab_time + hand_rmse + performance"
    # result_1 = linear_mixed_model(formula_1, main_dataframe_long, main_dataframe_long["participant"])
    # formula_2 = "nasa_tlx ~ aoi_list + aoi_other_object + hr + hrv + head_acc + performance"
    # result_2 = linear_mixed_model(formula_2, main_dataframe_long, main_dataframe_long["participant"])
    print(pingouin_test(main_dataframe_long))
    pass



