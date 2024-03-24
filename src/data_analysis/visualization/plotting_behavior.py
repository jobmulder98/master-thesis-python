from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.data_analysis.visualization.plotting_aoi import ray_direction_histogram_participant_condition
from src.data_analysis.visualization.plotting_head_movements import line_plot_head_movements_condition
from src.preprocessing.helper_functions.general_helpers import load_pickle, write_pickle
from src.data_analysis.visualization.plotting_ecg import plot_heart_rate_participant_condition

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]
hex_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
conditions = np.arange(1, 8)
participants = np.arange(1, 23)


def add_time_interval(time_interval_dict, tag, start, end):
    if tag not in time_interval_dict:
        time_interval_dict[tag] = []
    time_interval_dict[tag].append((start, end))
    return time_interval_dict


def handle_invalid_tags(df):
    previous_tag = None
    for index, row in df.iterrows():
        tag = row["focusObjectTag"]
        if tag == "Invalid":
            if previous_tag is not None:
                df.at[index, "focusObjectTag"] = previous_tag
        else:
            previous_tag = tag
    return df


def remove_time_interval_shorter_than(data: dict, threshold: float = 0.3, tag: str = None):
    if tag:
        # Only for one item of the dictionary
        data[tag] = [x for x in data[tag] if not x[1] - x[0] < threshold]
    else:
        # For the whole dictionary:
        for key, value in data.items():
            data[key] = [x for x in data[key] if not x[1] - x[0] < threshold]
    return data


def add_grabbing_time_intervals(dataframe, time_interval_dict):
    start_indices = dataframe[dataframe["isGrabbing"] & ~dataframe["isGrabbing"].shift(1, fill_value=False)].index
    end_indices = dataframe[~dataframe["isGrabbing"] & dataframe["isGrabbing"].shift(1, fill_value=False)].index

    if dataframe["isGrabbing"].iloc[-1]:
        end_indices = end_indices.append(pd.Index([len(dataframe["isGrabbing"]) - 1]))

    for start_index, end_index in zip(start_indices, end_indices):
        time_interval_dict = add_time_interval(time_interval_dict,
                                          "isGrabbing",
                                               dataframe["timeCumulative"].iloc[start_index],
                                               dataframe["timeCumulative"].iloc[end_index])

    return time_interval_dict


def behavior_dict(participant, condition):
    time_interval_dict = {}
    dataframe = create_clean_dataframe_hmd(participant, condition)
    dataframe["focusObjectTag"].replace({"notAssigned": "otherObject", "NPC": "otherObject"}, inplace=True)
    dataframe = handle_invalid_tags(dataframe)

    current_tag = None
    start_time = 0
    time = 0

    for index, row in dataframe.iterrows():
        tag = row["focusObjectTag"]
        time = row["timeCumulative"]

        if tag != current_tag:
            if current_tag is not None:
                time_interval_dict = add_time_interval(time_interval_dict, current_tag, start_time, time)
            current_tag = tag
            start_time = time

    if current_tag is not None:
        time_interval_dict = add_time_interval(time_interval_dict, current_tag, start_time, time)

    add_grabbing_time_intervals(dataframe, time_interval_dict)
    remove_time_interval_shorter_than(time_interval_dict, 0.4, "List")
    remove_time_interval_shorter_than(time_interval_dict, 1, "isGrabbing")

    return time_interval_dict


def save_behavior_dicts():
    all_dicts = {}
    for condidion in conditions:
        behavior_dicts = []
        for participant in participants:
            behavior_dicts.append(behavior_dict(participant, condidion))
        all_dicts[condidion] = behavior_dicts
    write_pickle("behavior_dicts.pickle", all_dicts)


def plot_behavior_chart(time_interval_dict, participant, condition, ax):
    tag_colors = {"MainShelf": "blue", "otherObject": "red", "Cart": "green", "List": "orange", "isGrabbing": "cyan"}
    key_order = ["otherObject", "Cart", "isGrabbing", "MainShelf", "List"]
    ax.set_yticks(range(len(key_order)))
    ax.set_yticklabels(key_order)
    for i, key in enumerate(key_order):
        intervals = time_interval_dict.get(key, [])
        for interval in intervals:
            bar_height = 0.3
            ax.barh(i, width=interval[1] - interval[0], left=interval[0], color=tag_colors.get(key, "gray"),
                    height=bar_height)

    ax.set_title(f"Behavior chart condition {condition_names[condition-1]}, participant {participant}".title(),
                 fontsize=11)


def behavior_head_movement_chart(participant, condition):
    time_interval_dict = behavior_dict(participant, condition)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    plot_behavior_chart(time_interval_dict, participant, condition, ax=ax1)
    line_plot_head_movements_condition(participant, condition)
    plt.tight_layout()
    plt.show()


def behavior_all_conditions(participant):
    # conditions = [1, 3, 5, 7]
    conditions = np.arange(1, 8)
    fig, axes = plt.subplots(len(conditions), 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': np.ones(len(conditions))})

    for i, condition in enumerate(conditions):
        time_interval_dict = behavior_dict(participant, condition)
        plot_behavior_chart(time_interval_dict, participant, condition, ax=axes[i])

    plt.tight_layout()
    plt.savefig(f"{DATA_DIRECTORY}/images/p{participant}.png")
    plt.savefig(f"{DATA_DIRECTORY}/images/behavior-charts-1234567/p{participant}.png")
    # plt.show()


def behavior_all_conditions_with_angle_histograms(participant):
    conditions = [1, 3, 7]
    fig, axes = plt.subplots(len(conditions), 2, figsize=(10, len(conditions) * 2), sharex='col',
                             gridspec_kw={'width_ratios': [3, 1], 'height_ratios': np.ones(len(conditions))})

    for i, condition in enumerate(conditions):
        time_interval_dict = behavior_dict(participant, condition)
        plot_behavior_chart(time_interval_dict, participant, condition, ax=axes[i, 0])
        ray_direction_histogram_participant_condition(participant, condition, ax=axes[i, 1])
        axes[i, 1].set_title(f"Ray Angle {condition_names[condition-1]}, participant {participant}".title(), fontsize=9)

        if i == len(conditions) - 1:
            axes[i, 0].set_xlabel("Time (seconds)")
            axes[i, 1].set_xlabel("Angle (degrees)")

    plt.tight_layout()
    plt.show()


def behavior_all_conditions_ecg(participant):
    conditions = [1, 3, 5, 7]
    fig, axes = plt.subplots(len(conditions)*2, 1, figsize=(8, len(conditions)*2), sharex=True,
                             gridspec_kw={'height_ratios': np.ones(len(conditions) * 2)})

    plt.subplots_adjust(hspace=0.2)

    for i, condition in enumerate(conditions):
        time_interval_dict = behavior_dict(participant, condition)
        plot_behavior_chart(time_interval_dict, participant, condition, ax=axes[i*2])
        plot_heart_rate_participant_condition(participant, condition, ax=axes[i*2+1], plot=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # behavior_head_movement_chart(4, 7)
    # for participant in participants:
    #     behavior_all_conditions(participant)
    behavior_all_conditions(4)
    # print(sns.color_palette("set2").as_hex())
    # behavior_all_conditions_with_angle_histograms(12)
    # save_behavior_dicts()
    # behavior_all_conditions_ecg(4)
    plt.show()
    pass
