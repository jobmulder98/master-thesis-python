from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import os

from src.data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.movements.hand_movements import (
    find_start_end_coordinates,
    rmse_hand_trajectory,
    mean_grab_time,
    mean_jerk,
)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]


def box_plot_hand_movements_rmse():
    if pickle_exists("box_plot_hand_movements_rmse.pickle"):
        hand_movement_rmse = load_pickle("box_plot_hand_movements_rmse.pickle")
    else:
        hand_movement_rmse = dict()
        for condition in conditions:
            rmse_condition = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                start_end_coordinates = find_start_end_coordinates(dataframe)
                rmse_trajectory = rmse_hand_trajectory(dataframe, start_end_coordinates)
                rmse_condition.append(rmse_trajectory)
            hand_movement_rmse[condition] = rmse_condition
        write_pickle("box_plot_hand_movements_rmse.pickle", hand_movement_rmse)
    fig, ax = plt.subplots()
    data = pd.DataFrame(hand_movement_rmse)

    # Participant 10 grabs object and puts it back, resulting in outlier. This removes the outlier
    data[1][data[1] > 0.7] = np.nan

    ax.set_title(f"RMSE of hand trajectory of product".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("RMSE")
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    plt.show()
    return


def box_plot_jerk():
    if pickle_exists("box_plot_jerk.pickle"):
        jerk = load_pickle("box_plot_jerk.pickle")
    else:
        jerk = dict()
        for condition in conditions:
            jerk_condition = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                start_end_coordinates = find_start_end_coordinates(dataframe)
                _mean_jerk = mean_jerk(dataframe, start_end_coordinates)
                jerk_condition.append(_mean_jerk)
            jerk[condition] = jerk_condition
        write_pickle("box_plot_jerk.pickle", jerk)
    fig, ax = plt.subplots()
    data = pd.DataFrame(jerk)

    ax.set_title(f"Mean Jerk of grabbing trajectories".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Jerk ($m/s^3$)")
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    plt.show()
    return


def box_plot_hand_movements_grab_time():
    if pickle_exists("box_plot_hand_movements_grab_time.pickle"):
        hand_movement_mean_grab_time = load_pickle("box_plot_hand_movements_grab_time.pickle")
    else:
        hand_movement_mean_grab_time = dict()
        for condition in conditions:
            mean_grab_times = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                start_end_coordinates = find_start_end_coordinates(dataframe)
                grab_time = mean_grab_time(start_end_coordinates)
                mean_grab_times.append(grab_time)
            hand_movement_mean_grab_time[condition] = mean_grab_times
        write_pickle("box_plot_hand_movements_grab_time.pickle", hand_movement_mean_grab_time)
    fig, ax = plt.subplots()
    data = pd.DataFrame(hand_movement_mean_grab_time)
    ax.set_title(f"Mean grab time of hand trajectory of product".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Time (s)")
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    plt.show()
    return


# box_plot_hand_movements_rmse()
# box_plot_hand_movements_grab_time()
box_plot_jerk()
