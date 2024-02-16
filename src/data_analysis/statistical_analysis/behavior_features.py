from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.data_analysis.visualization.plotting_behavior import behavior_dict, add_grabbing_time_intervals
from src.preprocessing.helper_functions.general_helpers import write_pickle, load_pickle, pickle_exists
from src.data_analysis.helper_functions.visualization_helpers import save_figure

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]
conditions = np.arange(1, 8)
participants = np.arange(1, 23)

# print(behavior_dict(4, 1))
# keys: otherObject, MainShelf, List, Cart, isGrabbing


def total_time_from_intervals(intervals):
    total_time = 0
    for start, end in intervals:
        total_time += (end - start)
    return total_time


def percentage_list_isgrabbing(participant, condition):
    """
    list looking shorter than 0.3 seconds are removed
    """
    dataframe = create_clean_dataframe_hmd(participant, condition)
    time_interval_dict = behavior_dict(participant, condition)
    time_interval_dict = add_grabbing_time_intervals(dataframe, time_interval_dict)
    overlaps = []

    for start1, end1 in time_interval_dict["List"]:
        for start2, end2 in time_interval_dict["isGrabbing"]:
            if start1 < end2 and end1 > start2:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlaps.append((overlap_start, overlap_end))

    total_overlap_time = total_time_from_intervals(overlaps)
    total_grabbing_time = total_time_from_intervals(time_interval_dict["isGrabbing"])

    percentage_overlap = total_overlap_time / total_grabbing_time * 100

    return percentage_overlap


def box_plot_percentage_list_isgrabbing():
    overlap_dictionary = load_pickle("percentage_list_isgrabbing.pickle")
    plot_dictionary = {}
    for condition in conditions:
        percentages = []
        for participant in participants:
            percentages.append(overlap_dictionary[condition][participant - 1])
        plot_dictionary[condition] = percentages
    data = pd.DataFrame(plot_dictionary)
    fig, ax = plt.subplots()
    plot_title = f"Percentage looking at List while Grabbing".title()
    ax.set_title(plot_title)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    # save_figure(f"boxplots/boxplot-behavior-grab.png")


def ratio_frequency_list_items():
    if not pickle_exists("ratio_frequency_list_items.pickle"):
        behavior_dictionaries = load_pickle("behavior_dicts.pickle")
        plot_dictionary = {}
        for condition in conditions:
            ratios = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                frequency_list = len(behavior_dictionaries[condition][participant - 1]["List"])
                frequency_items = dataframe["numberOfItemsInCart"].iloc[-1]
                ratios.append(frequency_list / frequency_items)
            plot_dictionary[condition] = ratios
        write_pickle("ratio_frequency_list_items.pickle", plot_dictionary)
    else:
        plot_dictionary = load_pickle("ratio_frequency_list_items.pickle")
    data = pd.DataFrame(plot_dictionary)
    fig, ax = plt.subplots()
    plot_title = f"Ratios frequency list:items".title()
    ax.set_title(plot_title)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Ratio (-)")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    # save_figure(f"boxplots/boxplot-behavior-ratio-frequency.png")



def ratio_time_list_items():
    if not pickle_exists("ratio_time_list_items.pickle"):
        behavior_dictionaries = load_pickle("behavior_dicts.pickle")
        plot_dictionary = {}
        for condition in conditions:
            ratios = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                time_list = total_time_from_intervals(behavior_dictionaries[condition][participant - 1]["List"])
                number_of_items = dataframe["numberOfItemsInCart"].iloc[-1]
                ratios.append(time_list / number_of_items)
            plot_dictionary[condition] = ratios
        write_pickle("ratio_time_list_items.pickle", plot_dictionary)
    else:
        plot_dictionary = load_pickle("ratio_time_list_items.pickle")
    data = pd.DataFrame(plot_dictionary)
    fig, ax = plt.subplots()
    plot_title = f"Ratios time list / items".title()
    ax.set_title(plot_title)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Seconds / item")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    # save_figure(f"boxplots/boxplot-behavior-ratio-time.png")




if __name__ == "__main__":
    box_plot_percentage_list_isgrabbing()
    ratio_frequency_list_items()
    ratio_time_list_items()
    pass

