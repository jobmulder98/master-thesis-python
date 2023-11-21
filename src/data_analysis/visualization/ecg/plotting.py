import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os
from matplotlib.widgets import CheckButtons

from preprocessing.ecg_eda.ecg.filtering import calculate_mean_heart_rate, calculate_heart_rate_variability
from preprocessing.helper_functions.general_helpers import load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
condition_names = ["no stimuli", "visual low", "visual high", "auditory low", "auditory high", "mental low", "mental high"]


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
    conditions = np.arange(1, 8)
    if outline_condition:
        opacity = np.ones(len(conditions)) * 0.2
        opacity[outline_condition-1] = 1
    else:
        opacity = np.ones(len(conditions))
    heart_rate_data = load_pickle(pickle_filename)
    times_dict, filtered_peaks_dict = heart_rate_data[0], heart_rate_data[1]
    for condition in np.arange(1, 8):
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
    plt.title('Heart Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (beats/min)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_per_condition(condition: int, pickle_filename: str):
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
    ax.set_xticklabels(heart_rates.keys())
    plt.show()
    return


def heart_rate_variability_boxplot(pickle_filename, participants, conditions):
    filtered_peaks = load_pickle(pickle_filename)
    times, peaks = filtered_peaks[0], filtered_peaks[1]
    heart_rate_variabilities = {}
    for condition in conditions:
        heart_rate_variability = []
        for participant in participants:
            hrv = calculate_heart_rate_variability(peaks[condition][participant-1])
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
    plt.show()
    return


conditions = np.arange(1, 8)
participants = np.arange(1, 22)
plot_average_heart_rate_all("ecg_data_filtered.pickle", outline_condition=None)
# plot_average_per_condition(3, "ecg_data_filtered.pickle")
# heart_rate_boxplot("ecg_data_unfiltered.pickle", participants, conditions)
# heart_rate_variability_boxplot("ecg_data_unfiltered.pickle", participants, conditions)
# plot_heart_rate_participant("ecg_data_filtered.pickle", 21)


