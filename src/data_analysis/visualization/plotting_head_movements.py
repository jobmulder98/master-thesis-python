from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

from preprocessing.hmd.movements.head_movements import head_stillness
from src.data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.movements.filtering_movements import filter_head_movement_data

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]


def box_plot_head_accelerations():
    if pickle_exists("head_acceleration_mean_results.pickle"):
        head_accelerations = load_pickle("head_acceleration_mean_results.pickle")
    else:
        head_accelerations = {}
        for condition in conditions:
            accelerations_condition = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                dataframe = filter_head_movement_data(dataframe)
                accelerations_condition.append(np.nanmean(dataframe["headMovementAccelerationFiltered"].values))
            head_accelerations[condition] = accelerations_condition
        write_pickle("head_acceleration_mean_results.pickle", head_accelerations)
    fig, ax = plt.subplots()
    data = pd.DataFrame(head_accelerations)
    ax.set_title(f"Mean head acceleration for all conditions".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Mean acceleration (m/s^2)")
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)


def box_plot_head_stillness():
    if pickle_exists("head_stillness.pickle"):
        head_accelerations = load_pickle("head_stillness.pickle")
    else:
        head_accelerations = {}
        for condition in conditions:
            head_stillness_condition = []
            for participant in participants:
                head_stillness_condition.append(head_stillness(participant, condition))
            head_accelerations[condition] = head_stillness_condition
        write_pickle("head_stillness.pickle", head_accelerations)

    fig, ax = plt.subplots()
    data = pd.DataFrame(head_accelerations)
    ax.set_title(f"Head stillness duration across all conditions".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Time (s)")
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)


def box_plot_idle_time(threshold=100):
    """
    Box plot of the total time in which accelerations are below a threshold
    in order to see how much people are in "Idle" state.

    Column of filtered head movements used: headMovementAccelerationFiltered
    """
    if pickle_exists(f"box_plot_idle_time_{threshold}.pickle"):
        head_movement_idle = load_pickle(f"box_plot_idle_time_{threshold}.pickle")
    else:
        head_movement_idle = {}
        for condition in conditions:
            idle_time_condition = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                dataframe = filter_head_movement_data(dataframe)
                mask = dataframe["headMovementAccelerationFiltered"] < threshold
                total_time = np.sum(dataframe.loc[mask, "deltaSeconds"])
                idle_time_condition.append(total_time)
            head_movement_idle[condition] = idle_time_condition
        write_pickle(f"box_plot_idle_time_{threshold}.pickle", head_movement_idle)
    fig, ax = plt.subplots()
    data = pd.DataFrame(head_movement_idle)
    ax.set_title(f"Total time idle head movement".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Time (s)")
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)


def box_plot_head_acceleration_peaks():
    if pickle_exists("head_acceleration_peaks_mean_results.pickle"):
        head_acceleration_peaks = load_pickle("head_acceleration_peaks_mean_results.pickle")
    else:
        head_acceleration_peaks = {}
        for condition in conditions:
            acc_peaks = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                dataframe = filter_head_movement_data(dataframe)
                head_movement_accelerations = dataframe["headMovementAccelerationFiltered"].values
                hm_peaks, _ = scipy.signal.find_peaks(head_movement_accelerations, height=200, distance=30)
                acc_peaks.append(len(hm_peaks))
            head_acceleration_peaks[condition] = acc_peaks
        write_pickle("head_acceleration_peaks_mean_results.pickle", head_acceleration_peaks)
    data = pd.DataFrame(head_acceleration_peaks)
    fig, ax = plt.subplots()
    ax.set_title(f"Number of peaks for participants in all conditions".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Number of peaks")
    sns.boxplot(data=data, ax=ax, palette="Set2")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)


def average_head_acceleration(outline_condition=None):
    opacity = increase_opacity_condition(outline_condition, len(conditions))
    common_times = np.arange(0, 122, step=0.01)
    if pickle_exists("head_acceleration_condition_averages.pickle"):
        data_dict = load_pickle("head_acceleration_condition_averages.pickle")
        common_times, avg_acceleration_per_condition = data_dict["common_times"], data_dict["average_data"]
    else:
        avg_acceleration_per_condition = {}
        for condition in conditions:
            total_acceleration_condition = []
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                dataframe = filter_head_movement_data(dataframe)
                head_movement_acceleration = dataframe["headMovementAcceleration"].values
                times = dataframe["timeCumulative"].values
                times[np.isnan(times)] = 0
                interpolated_acceleration = np.interp(common_times, times, head_movement_acceleration)
                total_acceleration_condition.append(interpolated_acceleration)
            stacked_arrays = np.stack(total_acceleration_condition, axis=0)
            average_array = np.mean(stacked_arrays, axis=0)
            avg_acceleration_per_condition[condition] = average_array
        # write_pickle("head_acceleration_condition_averages.pickle", {"common_times": common_times,
        #                                                              "average_data": avg_acceleration_per_condition})
    plt.figure(figsize=(10, 6))
    for condition in conditions:
        print(f"The mean of condition {condition} = {np.nanmean(avg_acceleration_per_condition[condition])}")
        plt.plot(common_times,
                 avg_acceleration_per_condition[condition],
                 label=f"Condition {condition}",
                 alpha=opacity[condition-1])
    plt.title("Average Head Movement Acceleration Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Acceleration")
    plt.legend()


def line_plot_head_movements_condition(participant, condition):
    dataframe = create_clean_dataframe_hmd(participant, condition)
    dataframe = filter_head_movement_data(dataframe)
    head_movement_acceleration = dataframe["headMovementAccelerationFiltered"].values
    times = dataframe["timeCumulative"].values
    times[np.isnan(times)] = 0
    hm_peaks, _ = scipy.signal.find_peaks(head_movement_acceleration, height=200, distance=30)

    plt.plot(times, head_movement_acceleration)
    plt.title(f"Head acceleration for participant {participant} in condition {condition}".title(),
              fontsize=9)
    plt.xlabel("Time (s)")
    plt.ylabel("Acc ($m/s^2$)")
    plt.scatter(times[hm_peaks], head_movement_acceleration[hm_peaks], color="red", marker=".")


def line_plot_accelerations_over_time(window_size: int = 10):
    time_window_dict = {condition: {} for condition in conditions}
    for condition in conditions:
        for participant in participants:
            df = create_clean_dataframe_hmd(participant, condition)
            df = filter_head_movement_data(df)
            for start_time in range(2, 122, window_size):
                end_time = start_time + window_size
                window_name = f"{start_time-2}-{end_time-2}"
                window_df = df[(df["timeCumulative"] >= start_time) & (df["timeCumulative"] < end_time)]
                average_acceleration = window_df["headMovementAcceleration"].mean()
                time_window_dict[condition].setdefault(window_name, []).append(average_acceleration)
    average_time_dict = {condition: {key: np.mean(values) for key, values in time_windows.items()} for
                         condition, time_windows in time_window_dict.items()}
    plt.figure(figsize=(10, 5))
    for condition in conditions:
        plt.plot(list(average_time_dict[condition].keys()), list(average_time_dict[condition].values()), marker='o',
                 label=f'Condition {condition}')
    plot_title = f"Mean head acceleration for all conditions in windows of {window_size} seconds".title()
    plt.title(plot_title)
    plt.xlabel("Time window (s)")
    plt.ylabel("Mean acceleration (m/s^2)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()


def line_plot_acceleration_peaks_over_time(window_size: int = 10):
    if pickle_exists("line_plot_acceleration_peaks_over_time.pickle"):
        peaks_dict = load_pickle("line_plot_acceleration_peaks_over_time.pickle")
    else:
        time_window_dict = {condition: {} for condition in conditions}
        for condition in conditions:
            for participant in participants:
                df = create_clean_dataframe_hmd(participant, condition)
                df = filter_head_movement_data(df)
                hm_peaks, _ = scipy.signal.find_peaks(df["headMovementAccelerationFiltered"], height=200, distance=30)
                for start_time in range(2, 122, window_size):
                    end_time = start_time + window_size
                    window_name = f"{start_time-2}-{end_time-2}"
                    start_index = df[df["timeCumulative"] >= start_time].index.min()
                    end_index = df[df["timeCumulative"] < end_time].index.max()
                    peaks_in_window = [peak for peak in hm_peaks if start_index <= peak <= end_index]
                    time_window_dict[condition].setdefault(window_name, []).append(len(peaks_in_window))
        peaks_dict = {condition: {key: np.mean(values) for key, values in time_windows.items()} for
                      condition, time_windows in time_window_dict.items()}
        write_pickle("line_plot_acceleration_peaks_over_time.pickle", peaks_dict)
    plt.figure(figsize=(10, 5))
    for condition in conditions:
        plt.plot(list(peaks_dict[condition].keys()), list(peaks_dict[condition].values()), marker='o',
                 label=f'Condition {condition}')
    plot_title = f"Average number of peaks in accelerations for conditions in windows of {window_size} seconds".title()
    plt.title(plot_title)
    plt.xlabel("Time window (s)")
    plt.ylabel("Number of peaks (>200 m/s^2)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()


if __name__ == "__main__":
    # box_plot_head_accelerations()
    # box_plot_head_acceleration_peaks()
    # box_plot_idle_time(threshold=75)
    # average_head_acceleration(6)
    # line_plot_head_movements_condition(20, 4)
    # line_plot_accelerations_over_time(10)
    # line_plot_acceleration_peaks_over_time(10)
    box_plot_head_stillness()
    plt.show()
    pass
