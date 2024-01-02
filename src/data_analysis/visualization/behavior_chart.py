from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.data_analysis.visualization.plotting_aoi import ray_direction_histogram_participant_condition
from src.data_analysis.visualization.plotting_head_movements import line_plot_head_movements_condition

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]
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


def plot_behavior_chart(time_interval_dict, participant, condition, ax):
    tag_colors = {"MainShelf": "blue", "otherObject": "red", "Cart": "green", "List": "orange", "isGrabbing": "cyan"}

    for i, (key, intervals) in enumerate(time_interval_dict.items()):
        for interval in intervals:
            bar_height = 0.3
            ax.barh(key, width=interval[1] - interval[0], left=interval[0], color=tag_colors.get(key, "gray"),
                    height=bar_height)

    ax.set_title(f"Behavior chart condition {condition_names[condition-1]}, participant {participant}".title(),
                 fontsize=9)


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

    return time_interval_dict


def behavior_head_movement_chart(participant, condition):
    time_interval_dict = behavior_dict(participant, condition)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    plot_behavior_chart(time_interval_dict, participant, condition, ax=ax1)
    line_plot_head_movements_condition(participant, condition)
    plt.tight_layout()
    plt.show()


def behavior_all_conditions(participant):
    conditions = [1, 3, 7]
    fig, axes = plt.subplots(len(conditions), 1, figsize=(8, len(conditions)*2), sharex=True,
                             gridspec_kw={'height_ratios': np.ones(len(conditions))})

    for i, condition in enumerate(conditions):
        time_interval_dict = behavior_dict(participant, condition)
        plot_behavior_chart(time_interval_dict, participant, condition, ax=axes[i])

    plt.tight_layout()
    plt.show()


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


# behavior_head_movement_chart(4, 7)
# behavior_all_conditions(11)
behavior_all_conditions_with_angle_histograms(4)
