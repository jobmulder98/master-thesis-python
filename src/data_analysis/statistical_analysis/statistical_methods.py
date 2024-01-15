import scipy.stats
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import pingouin as pg

import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
import statsmodels.formula.api as smf

from src.data_analysis.helper_functions.data_helpers import obtain_feature_data
from src.preprocessing.helper_functions.general_helpers import write_pickle, load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)


def anova(data: dict):
    data_values = list(data.values())
    return stats.f_oneway(*data_values)


def repeated_measures_anova(dataframe, measure):
    aov_rm = AnovaRM(dataframe, depvar=measure, subject='participant', within=['condition'])
    result = aov_rm.fit()
    return result


def linear_mixed_model(formula, dataframe, groups):
    lmm = smf.mixedlm(formula, dataframe, groups=groups)
    lmm_result = lmm.fit()
    return lmm_result


def pairwise_tukey_test(data: dict):
    value_list = []
    key_list = []
    for key, values in data.items():
        value_list.extend(values)
        key_list.extend([key] * len(values))
    tukey = pairwise_tukeyhsd(value_list, key_list, alpha=0.05)
    return tukey


def pingouin_test(dataframe, dv='nasa_tlx', within="condition", subject="participant"):
    result_mauchly = pg.sphericity(dataframe, dv=dv, within=within, subject=subject)
    return result_mauchly


def post_hoc_test(feature_1: list, feature_2: list):
    return stats.ttest_ind(feature_1, feature_2)


if __name__ == '__main__':
    """
    measures1 = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
        "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"]
        
    measures2 = ["overlap_grab_list", "ratio_frequency_list_items", "ratio_time_list_items"]
    """

    measures1 = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
                 "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"]

    main_dataframe = load_pickle("main_dataframe.pickle")
    main_dataframe_long = load_pickle("main_dataframe_long.pickle")

    # measure = "hrv"

    # Repeated Measures ANOVA Test
    for measure in measures1:
        rm_anova_result = repeated_measures_anova(main_dataframe_long, measure)
        # print("Repeated Measures ANOVA Summary:")
        # print(rm_anova_result.summary())
        p_value = rm_anova_result.anova_table["Pr > F"]["condition"]
        print(f"\n ANOVA {measure.capitalize()} P-VALUE: {p_value}")

        if p_value < 0.05:
            # posthoc = pg.pairwise_tukey(data=main_dataframe_long, dv=measure, between="condition")
            # print(posthoc)
            pvalues = ""
            for i in range(2, 8):
                condition_1_data = main_dataframe_long[main_dataframe_long['condition'] == 1][measure]
                condition_i_data = main_dataframe_long[main_dataframe_long['condition'] == i][measure]
                statistic, p_value = scipy.stats.ttest_ind(condition_1_data, condition_i_data)
                pvalues += f"{p_value.round(3)} & "
            print(pvalues)


    # Friedman Test
    # for measure in measures1:
    #     groups = [main_dataframe_long[measure][main_dataframe_long['condition'] == condition] for condition in
    #               main_dataframe_long['condition'].unique()]
    #     _, p_value_friedman = stats.friedmanchisquare(*groups)
    #     print(f"\nFriedman Test Result for {measure}:")
    #     print(f"P-value: {p_value_friedman}")


    pass



