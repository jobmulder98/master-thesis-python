from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import os

from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.data_analysis.statistical_analysis.statistical_checks import calculate_cohens_d

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]


def effect_size_bar_plot(data, opacity):
    measures = list(data.keys())
    values = np.array(list(data.values()))
    opacities = np.array(list(opacity.values()))
    num_measures, num_conditions = values.shape

    bar_width = 0.8 / num_conditions
    index = np.arange(num_measures)

    fig, ax = plt.subplots(figsize=(14, 5))

    for j in range(num_conditions):
        offset = j * bar_width - 0.4 + bar_width / 2
        for i, measure in enumerate(measures):
            val = values[i, j]
            op = opacities[i, j]
            color = "blue"
            text_offset = 0.15 if val >= 0 else -0.5
            ax.bar(index[i] + offset, val, bar_width, color=color, edgecolor="black", alpha=op)
            ax.text(index[i] + offset, val + text_offset, j+2, ha='center', va='bottom', fontsize=8, color="black")

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Effect Size")
    ax.set_title("Bar plot with effect sizes. Bars are dark blue when $p<0.05$".title())
    ax.set_xticks(index)
    ax.set_xticklabels(measures)

    legend_text = "Bar numbers:\n2 = Visual Low\n3 = Visual High\n4 = Auditory Low\n5 = Auditory High\n6 = Mental Low\n7 = Mental High"
    ax.text(0.05, 0.95, legend_text, transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.show()


# Example usage:
# measure_dict = {
#     "Total Gaze Time Spent on Cart": [0.062, -0.374, -0.390, -0.199, -0.802, -0.722],
#     "Total Gaze Time Spent on List": [0.129, 0.062, 0.549, 0.241, 1.064, 2.216],
#     "Total Gaze Time Spent on Main Shelf": [-0.518, -1.102, -0.449, -0.131, -0.738, -2.487],
#     "Total Gaze Time Spent on Other Object": [1.015, 2.947, -0.020, 0.142, -0.017, 0.661],
#     "Mean Heart Rate": [0.004, -0.298, -0.263, 0.149, 0.822, 0.762],
#     "Mean Heart Rate Variability": [0.125, 0.111, -0.046, 0.184, -0.622, -0.116],
#     "Mean Jerk of Hand Trajectory": [-3.133, -4.784, -0.400, 0.014, 0.290, -0.182],
#     "Total Time Head in Stationary State": [1.105, 1.751, -0.003, 0.245, 0.118, 2.945],
#     "Performance": [0.333, -0.152, 0.099, -0.559, -0.269, -3.006],
#     "NASA-TLX Score": [-0.081, 0.569, 0.236, 1.389, 1.539, 4.706]
# }
measure_dict = {
    "The percentage of time\n viewing the list while\n grabbing an object": [-0.157, -0.216, 0.067, 0.186, 0.984, -0.985],
    "The ratio of list viewing\n frequency to collected\n items": [-0.136, 0.078, 0.054, 0.1432, 0.454, 3.813],
    "The average number of \nseconds looking at the list\n per item": [-0.012, 0.081, 0.220, 0.425, 0.646, 4.560]
}

not_sig = 0.25
significant = 0.5
# bold_values_dict = {
#     "Total Gaze Time Spent on Cart": [not_sig, not_sig, not_sig, not_sig, not_sig, not_sig],
#     "Total Gaze Time Spent on List": [not_sig, not_sig, not_sig, not_sig, significant, significant],
#     "Total Gaze Time Spent on Main Shelf": [not_sig, significant, not_sig, not_sig, not_sig, significant],
#     "Total Gaze Time Spent on Other Object": [significant, significant, not_sig, not_sig, not_sig, not_sig],
#     "Mean Heart Rate": [not_sig, not_sig, not_sig, not_sig, not_sig, not_sig],
#     "Mean Heart Rate Variability": [not_sig, not_sig, not_sig, not_sig, not_sig, not_sig],
#     "Mean Jerk of Hand Trajectory": [significant, significant, not_sig, not_sig, not_sig, not_sig],
#     "Total Time Head in Stationary State": [significant, significant, not_sig, not_sig, not_sig, significant],
#     "Performance": [not_sig, not_sig, not_sig, not_sig, not_sig, significant],
#     "NASA-TLX Score": [not_sig, not_sig, not_sig, significant, significant, significant]
# }
bold_values_dict = {
    "The percentage of time viewing the list while grabbing an object": [0.25, 0.25, 0.25, 0.25, 0.8, 0.8],
    "The ratio of list viewing frequency to collected items": [0.25, 0.25, 0.25, 0.25, 0.25, 0.8],
    "The average number of seconds looking at the list per item": [0.25, 0.25, 0.25, 0.25, 0.25, 0.8]
}

effect_size_bar_plot(measure_dict, bold_values_dict)
