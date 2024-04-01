import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os
import seaborn as sns
import pandas as pd

from data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from preprocessing.helper_functions.general_helpers import load_pickle, write_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
EDA_SAMPLE_RATE = int(os.getenv("EDA_SAMPLE_RATE"))
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]
conditions = np.arange(1, 8)
participants = np.arange(1, 23)
# participants = [3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]


def box_plot_normalized_mean_median(plot_mean=True):
    """
    Plots a box plot with the normalized means of all conditions.

    params: plot_mean = True -> plots the means,
                        False -> plots the medians.
    """
    feature = "means_norm" if plot_mean else "medians_norm"
    feature_name = "mean" if plot_mean else "median"
    normalized_data = load_pickle("eda_normalized_means_medians.pickle")
    plotting_dictionary = {}
    for condition in range(1, 8):
        means_list = []
        for participant, data in normalized_data.items():
            means_list.append(data[feature][condition - 1])
        plotting_dictionary[condition] = means_list
    fig, ax = plt.subplots()
    data = pd.DataFrame(plotting_dictionary)
    ax.set_title(f"Normalized {feature_name} of skin conductance for all conditions".title())
    ax.set_xlabel("Condition")
    ax.set_xticklabels(condition_names)
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Normalized Value For Skin Conductance".title())
    sns.boxplot(data=data, ax=ax, palette="Set1")
    sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    # plt.show()


def box_plot_normalized_signal(pickle_filename: str, outline_condition):
    """
    The difference between this and the previous function is the standardisation. Here the signal is standardised,
    while in the previous function the means are standardised.
    """
    opacity = increase_opacity_condition(outline_condition, len(conditions))
    eda_dictionary = load_pickle(pickle_filename)
    synchronized_times = load_pickle("synchronized_times.pickle")
    for condition in conditions:
        eda_signal = []
        time_data = []
        for participant in participants:
            start_index, end_index = synchronized_times[participant][condition]
            start_index = int(start_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
            end_index = int(end_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
            _, filtered_signal, tonic_signal, phasic_signal = eda_dictionary[participant]
            # mean_value = np.mean(filtered_signal)
            # standard_deviation = np.std(filtered_signal)
            # normalized_data = [((x - mean_value) / standard_deviation) for x in filtered_signal]
            signal_condition = phasic_signal[start_index:end_index]

            eda_signal.append(signal_condition)
            eda_signal_length = len(signal_condition)
            time_data.append(np.arange(eda_signal_length) / EDA_SAMPLE_RATE)
        common_x = np.linspace(0, 122, 1000)
        interp_y = [np.interp(common_x, x_values, y_values) for x_values, y_values in zip(time_data, eda_signal)]
        average_eda = np.mean(interp_y, axis=0)
        plt.plot(common_x,
                 average_eda,
                 linestyle='-',
                 linewidth=2,
                 alpha=opacity[condition - 1],
                 label=condition_names[condition - 1])
    plt.title('Average Skin Conductance Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Standardized EDA Signal')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_skin_conductance(pickle_filename: str, outline_condition):
    opacity = increase_opacity_condition(outline_condition, len(conditions))
    eda_dictionary = load_pickle(pickle_filename)
    synchronized_times = load_pickle("synchronized_times.pickle")
    for condition in conditions:
        eda_signal = []
        time_data = []
        for participant in participants:
            start_index, end_index = synchronized_times[participant][condition]
            start_index = int(start_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
            end_index = int(end_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
            _, filtered_signal, tonic_signal, phasic_signal = eda_dictionary[participant]
            # mean_value = np.mean(filtered_signal)
            # standard_deviation = np.std(filtered_signal)
            # normalized_data = [((x - mean_value) / standard_deviation) for x in filtered_signal]
            signal_condition = phasic_signal[start_index:end_index]

            eda_signal.append(signal_condition)
            eda_signal_length = len(signal_condition)
            time_data.append(np.arange(eda_signal_length) / EDA_SAMPLE_RATE)
        common_x = np.linspace(0, 122, 1000)
        interp_y = [np.interp(common_x, x_values, y_values) for x_values, y_values in zip(time_data, eda_signal)]
        average_eda = np.mean(interp_y, axis=0)
        plt.plot(common_x,
                 average_eda,
                 linestyle='-',
                 linewidth=2,
                 alpha=opacity[condition - 1],
                 label=condition_names[condition - 1])
    plt.title('Average Skin Conductance Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Standardized EDA Signal')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_skin_conductance_per_condition(condition: int, pickle_filename: str):
    eda_dictionary = load_pickle(pickle_filename)
    synchronized_times = load_pickle("synchronized_times.pickle")
    eda_signal = []
    time_data = []
    for participant in participants:
        start_index, end_index = synchronized_times[participant][condition]
        start_index = int(start_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
        end_index = int(end_index / ECG_SAMPLE_RATE * EDA_SAMPLE_RATE)
        _, filtered_signal, tonic_signal, phasic_signal = eda_dictionary[participant]
        signal_condition = tonic_signal[start_index:end_index]
        eda_signal.append(signal_condition)
        eda_signal_length = len(signal_condition)
        time = np.arange(eda_signal_length) / EDA_SAMPLE_RATE
        time_data.append(time)
        plt.plot(time, signal_condition, linewidth=1, linestyle='-', color="blue", alpha=0.5)
    common_x = np.linspace(0, 122, 1000)
    interp_y = [np.interp(common_x, x_values, y_values) for x_values, y_values in zip(time_data, eda_signal)]
    average_eda = np.mean(interp_y, axis=0)
    plt.plot(common_x,
             average_eda,
             linestyle='-',
             linewidth=2,
             color="red",
             label=condition_names[condition - 1])
    plt.title('Skin conductance over time')
    plt.xlabel('Time (s)')
    plt.ylabel('Standardized EDA Signal')
    plt.legend()
    plt.grid(True)
    plt.show()


box_plot_normalized_mean_median(plot_mean=True)
# plot_average_skin_conductance("eda_signal_filtered_normalized.pickle", outline_condition=None)
# plot_average_skin_conductance_per_condition(4, "eda_signal_filtered_normalized.pickle")
