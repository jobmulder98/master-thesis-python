import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os
from matplotlib.widgets import CheckButtons

from data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from preprocessing.ecg_eda.ecg.filtering import (calculate_mean_heart_rate,
                                                 calculate_rmssd,
                                                 interpolate_nan_values)
from preprocessing.helper_functions.general_helpers import load_pickle
load_dotenv()

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]
conditions = np.arange(1, 8)


participants = np.arange(1, 22)


def heart_rate_boxplot(pickle_filename, participants, conditions):
    filtered_peaks = load_pickle(pickle_filename)
    times, peaks = filtered_peaks[0], filtered_peaks[1]
    heart_rates = {}
    for condition in conditions:
        heart_rate = []
        for participant in participants:
            hr = calculate_mean_heart_rate(times[condition][participant-1], peaks[condition][participant-1])
            heart_rate.append(hr)
        filtered_data = [x for x in heart_rate if x is not None]
        heart_rates[condition] = filtered_data
    fig, ax = plt.subplots()
    ax.set_title("Heart Rate")
    ax.set_xlabel("Condition")
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel("Heart Rate (beats/min)")
    ax.boxplot(heart_rates.values())
    ax.set_xticklabels(condition_names)
    plt.show()
    return


def heart_rate_variability_boxplot(pickle_filename, participants, conditions):
    filtered_peaks = load_pickle(pickle_filename)
    times, peaks = filtered_peaks[0], filtered_peaks[1]
    heart_rate_variabilities = {}
    for condition in conditions:
        heart_rate_variability = []
        for participant in participants:
            hrv = calculate_rmssd(peaks[condition][participant-1])
            heart_rate_variability.append(hrv)
        filtered_data = [x for x in heart_rate_variability if x is not None]
        heart_rate_variabilities[condition] = filtered_data
    fig, ax = plt.subplots()
    ax.set_title("Heart Rate Variability")
    ax.set_xlabel("Condition")
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel("Heart Rate Variability (beats/min)")
    ax.boxplot(heart_rate_variabilities.values())
    ax.set_xticklabels(condition_names)
    # plt.show()
    return


def plot_heart_rate_participant(pickle_filename: str, participant: int):
    fig, ax = plt.subplots()
    heart_rate_data = load_pickle(pickle_filename)
    times_dict, filtered_peaks_dict = heart_rate_data[0], heart_rate_data[1]
    condition_visibility = {name: True for name in condition_names}

    def update_visibility(label):
        condition_visibility[label] = not condition_visibility[label]
        update_plot()

    def update_plot():
        ax.clear()
        ax.set_title("Heart Rate Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart Rate (bpm)")
        for condition_name in condition_names:
            if condition_visibility[condition_name]:
                condition_index = condition_names.index(condition_name) + 1
                times = times_dict[condition_index][participant-1]
                filtered_peaks = filtered_peaks_dict[condition_index][participant-1]
                filtered_data = [(xi, yi) for xi, yi in zip(times, filtered_peaks) if xi is not None and yi is not None]
                times, filtered_peaks = zip(*filtered_data)
                ax.plot(times, filtered_peaks, marker='.', linestyle='-', label=condition_name)
        ax.legend()
        ax.grid(True)
        plt.draw()
    checkboxes_axes = plt.axes([0.05, 0.1, 0.1, 0.2])
    checkboxes = CheckButtons(checkboxes_axes, condition_names, actives=[True] * len(condition_names))
    checkboxes.on_clicked(update_visibility)
    update_plot()
    plt.show()


def plot_average_heart_rate_all(pickle_filename: str, outline_condition=None):
    opacity = increase_opacity_condition(outline_condition, len(conditions))
    heart_rate_data = load_pickle(pickle_filename)
    times_dict, filtered_peaks_dict = heart_rate_data[0], heart_rate_data[1]
    for condition in conditions:
        times = times_dict[condition]
        filtered_peaks = filtered_peaks_dict[condition]
        filtered_data = [(xi, yi) for xi, yi in zip(times, filtered_peaks) if xi is not None and yi is not None]
        times, filtered_peaks = zip(*filtered_data)
        common_x = np.linspace(min(min(sublist) for sublist in times), max(max(sublist) for sublist in times), 1000)
        interp_y = [np.interp(common_x, x_values, y_values) for x_values, y_values in zip(times, filtered_peaks)]
        average_y = np.mean(interp_y, axis=0)
        plt.plot(common_x,
                 average_y,
                 linestyle='-',
                 linewidth=2,
                 alpha=opacity[condition-1],
                 label=condition_names[condition-1])
    # plt.axvspan(0, 5, facecolor="black", alpha=0.3)
    # plt.axvspan(31, 36, facecolor="black", alpha=0.3)
    # plt.axvspan(62, 67, facecolor="black", alpha=0.3)
    # plt.axvspan(93, 98, facecolor="black", alpha=0.3)
    plt.title('Heart Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (beats/min)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_heart_rate_variability_all(pickle_filename: str, outline_condition=None):
    opacity = increase_opacity_condition(outline_condition, len(conditions))
    heart_rate_data = load_pickle(pickle_filename)
    times_dict, filtered_peaks_dict = heart_rate_data[0], heart_rate_data[1]
    for condition in conditions:
        filtered_peaks = filtered_peaks_dict[condition]
        times = times_dict[condition]
        filtered_data = [(xi, yi) for xi, yi in zip(times, filtered_peaks) if xi is not None and yi is not None]
        times, filtered_peaks = zip(*filtered_data)
        hrv_data = []
        time_data = []

        for participant_data, participant_time in zip(filtered_peaks, times):
            rpeaks_interpolated = interpolate_nan_values(participant_data)
            rr_intervals = [60 / (hr / ECG_SAMPLE_RATE) for hr in rpeaks_interpolated]
            differences_between_consecutive_intervals = np.diff(rr_intervals)
            hrv_data.append(np.abs(differences_between_consecutive_intervals**2))
            time_data.append(participant_time[1:])
        common_x = np.linspace(min(min(sublist) for sublist in times), max(max(sublist) for sublist in times), 1000)
        interp_y = [np.interp(common_x, x_values, y_values) for x_values, y_values in zip(time_data, hrv_data)]
        average_y = np.mean(interp_y, axis=0)
        plt.plot(common_x,
                 average_y,
                 linestyle='-',
                 linewidth=2,
                 # color=colors[condition-1],
                 alpha=opacity[condition - 1],
                 label=condition_names[condition - 1])

    plt.title('Average HRV Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Average HRV (ms^2)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_heart_rate_per_condition(condition: int, pickle_filename: str):
    heart_rate_data = load_pickle(pickle_filename)
    times_dict, filtered_peaks_dict = heart_rate_data[0], heart_rate_data[1]
    times = times_dict[condition]
    filtered_peaks = filtered_peaks_dict[condition]
    filtered_data = [(xi, yi) for xi, yi in zip(times, filtered_peaks) if xi is not None and yi is not None]
    times, filtered_peaks = zip(*filtered_data)
    common_x = np.linspace(min(min(sublist) for sublist in times), max(max(sublist) for sublist in times), 1000)
    interp_y = [np.interp(common_x, x_values, y_values) for x_values, y_values in zip(times, filtered_peaks)]
    average_y = np.mean(interp_y, axis=0)
    for x_values, y_values in zip(times, filtered_peaks):
        plt.plot(x_values, y_values, marker='.', linestyle='-', color="grey", alpha=0.5)
    plt.plot(common_x, average_y, color='red', linestyle='-', linewidth=2, label='Average')
    plt.title('Heart Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (beats/min)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_heart_rate_participant_condition(participant: int, condition: int, ax=None, plot=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 1))
        ax.set_title("Heart Rate Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BPM")
    heart_rate_data = load_pickle("ecg_data_filtered.pickle")
    times_dict, filtered_peaks_dict = heart_rate_data[0], heart_rate_data[1]
    times = times_dict[condition][participant-1]
    filtered_peaks = filtered_peaks_dict[condition][participant-1]
    filtered_data = [(xi, yi) for xi, yi in zip(times, filtered_peaks) if xi is not None and yi is not None]
    times, filtered_peaks = zip(*filtered_data)
    ax.plot(times, filtered_peaks, marker='.', linestyle='-')
    if plot:
        plt.show()


if __name__ == "__main__":
    # heart_rate_boxplot("ecg_data_unfiltered.pickle", participants, conditions)
    # heart_rate_variability_boxplot("ecg_data_unfiltered.pickle", participants, conditions)
    # plot_average_heart_rate_all("ecg_data_filtered.pickle", outline_condition=[1, 7])
    # plot_heart_rate_variability_all("ecg_data_filtered.pickle", outline_condition=[5, 6, 7])
    # plot_average_heart_rate_per_condition(1, "ecg_data_filtered.pickle")
    # plot_heart_rate_participant("ecg_data_filtered.pickle", 7)
    # plot_heart_rate_participant_condition(4, 7)
    pass

