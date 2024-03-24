from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

from src.data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.performance.filtering import (
    count_errors,
    seconds_per_item,
    product_lists,
    n_back_performance_dataframe,
)

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]


def speed_accuracy_plot():
    if pickle_exists("performance_dataframe.pickle"):
        plotting_dataframe = load_pickle("performance_dataframe.pickle")
        plotting_dataframe["performance"] = [60 / x if x != 0 else 0 for x in plotting_dataframe["performance"]]
    else:
        speed_participants = []
        errors_participants = []
        condition_values = []
        for condition in conditions:
            for participant in participants:
                dataframe = create_clean_dataframe_hmd(participant, condition)
                speed_participants.append(seconds_per_item(dataframe))
                correct_order, participant_picks = product_lists(dataframe)
                errors_participants.append(count_errors(correct_order, participant_picks, participant, condition))
                condition_values.append(condition)
        plotting_dataframe = pd.DataFrame({"performance": speed_participants,
                                           "condition": condition_values,
                                           "errors": errors_participants})
        write_pickle("performance_dataframe.pickle", plotting_dataframe)

    fig, ax = plt.subplots()
    fig.autofmt_xdate(rotation=30)
    colors = {0: "grey", 1: "red", 2: "green", 3: "cyan", 4: "blue"}
    nan_color = "grey"
    plotting_dataframe["color"] = plotting_dataframe["errors"].apply(lambda x: colors[x] if x in colors else nan_color)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[key], markersize=10, label=str(key)) for key in
        colors]
    ax.scatter(
        plotting_dataframe["condition"] + np.random.uniform(-0.1, 0.1, len(plotting_dataframe)),
        plotting_dataframe["performance"],
        marker="o",
        color=plotting_dataframe["color"],
        alpha=0.75,
    )
    ax.set_title(f"Speed-Accuracy across all conditions".title())
    ax.set_xlabel("Condition")
    ax.set_xticks(conditions)
    ax.set_xticklabels(condition_names)
    ax.set_ylabel("Items per Minute")
    ax.legend(handles=legend_elements, title='Errors', loc='best')
    plt.show()


def n2_back_speed_accuracy_plot():
    plotting_dataframe = n_back_performance_dataframe()
    plotting_dataframe["performance"] = [60 / x if x != 0 else 0 for x in plotting_dataframe["performance"]]
    errors = [0, 1, 0, 1, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 0]  # hardcoded, should fix later
    plotting_dataframe["errors"] = errors

    if plotting_dataframe.empty:
        return "Empty dataframe; no pickle for dataframe found."

    fig, ax = plt.subplots()
    colors = {0: "grey", 1: "red", 2: "green"}
    nan_color = "grey"
    plotting_dataframe["color"] = plotting_dataframe["errors"].apply(lambda x: colors[x] if x in colors else nan_color)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[key], markersize=10, label=str(key)) for key in
        colors]

    a, b = np.polyfit(plotting_dataframe["n_back_correct"], plotting_dataframe["performance"], 1)
    plt.plot(plotting_dataframe["n_back_correct"], a*plotting_dataframe["n_back_correct"]+b, alpha=0.5)
    ax.scatter(
        plotting_dataframe["n_back_correct"],
        plotting_dataframe["performance"],
        marker="o",
        color=plotting_dataframe["color"],
    )
    print(stats.pearsonr(plotting_dataframe["n_back_correct"], plotting_dataframe["performance"]))
    ax.set_title("2-back task accuracy vs. speed".title())
    ax.set_xlabel("Accuracy (correct 2-back task answers, max 30)")
    ax.set_ylabel("Speed (items/minute)")
    ax.legend(handles=legend_elements, title='Product Errors', loc='best')
    plt.show()


def n2_back_speed_accuracy_plot_participant_type():
    plotting_dataframe = n_back_performance_dataframe()

    ratio_frequency_list_items = load_pickle("ratio_frequency_list_items.pickle")
    ratio_time_list_items = load_pickle("ratio_time_list_items.pickle")

    colors_1 = ['red' if val > np.median(ratio_frequency_list_items[7]) else 'blue' for val in ratio_frequency_list_items[7]]
    colors_2 = ['red' if val > np.median(ratio_time_list_items[7]) else 'blue' for val in ratio_time_list_items[7]]

    if plotting_dataframe.empty:
        return "Empty dataframe; no pickle for dataframe found."

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True,
                           gridspec_kw={'height_ratios': np.ones(2)})

    ax1.scatter(
        plotting_dataframe["n_back_correct"],
        plotting_dataframe["performance"],
        marker="o",
        c=colors_1,
    )
    ax2.scatter(
        plotting_dataframe["n_back_correct"],
        plotting_dataframe["performance"],
        marker="o",
        c=colors_2,
    )
    print(stats.pearsonr(plotting_dataframe["n_back_correct"], plotting_dataframe["performance"]))
    ax1.set_title("2-back task accuracy vs. speed".title())
    ax2.set_xlabel("correct 2-back task answers (max 30)")
    ax1.set_ylabel("speed (seconds/item)")
    ax2.set_ylabel("speed (seconds/item)")

    a, b = np.polyfit(plotting_dataframe["n_back_correct"], plotting_dataframe["performance"], 1)
    ax1.plot(plotting_dataframe["n_back_correct"], a * plotting_dataframe["n_back_correct"] + b, alpha=0.5)
    ax2.plot(plotting_dataframe["n_back_correct"], a * plotting_dataframe["n_back_correct"] + b, alpha=0.5)

    legend_elements = [mlines.Line2D([], [], marker='o', linestyle='None', color='blue', label='< median', markersize=10),
                         mlines.Line2D([], [], marker='o', linestyle='None',  color='red', label='>= median', markersize=10)]
    ax1.legend(handles=legend_elements, title="Frequency of list:items-collected")
    ax2.legend(handles=legend_elements, title="Time spent at list / items-collected")
    plt.show()


def performance_nasa_tlx_3d_plot():
    performance_df = n_back_performance_dataframe()
    performance_speed = performance_df["performance"]
    performance_n_back = performance_df["n_back_correct"]

    main_dataframe = load_pickle("main_dataframe.pickle")
    nasa_tlx_condition_7 = main_dataframe["nasa_tlx_7"].values
    nasa_tlx_ratio = [3.05, 12.6, 4.19, 6.09, 3.32, 2.17, 1.39, 3.94, 1.87, 15.33, 3.19, 2.08, 2.54, 1.2, 4.64, 3.15,
                      3.97, 29.56, 4.69, 4.27, 3.98, 3.32]
    hr_ratio = [2.898446599828, 7.612309689955509, 8.87761617097111, 15.676080589967029, 3.697598062162072, 4.529166613939404, 10.765099249447136, 12.817840314420362, 10.220098779271435, 53.82780235911642, 11.856086010495309, 17.945457800143537, 11.81345868381989, 9.414911018971967, 6.591877076073444, 10.405475605944146, 31.14721485411141, 14.417208165769955, 22.613195565813147, 27.198569513080596, 8.577950940494347, 7.5000000]
    # print(np.median(hr_ratio))
    colors = ['red' if val > np.median(hr_ratio) else 'blue' for val in hr_ratio]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(performance_speed, performance_n_back, hr_ratio, c=colors)
    ax.set_xlabel('performance speed/item')
    ax.set_ylabel('performance n-back correct')
    ax.set_zlabel('hr percentage')
    plt.show()


# speed_accuracy_plot()
# n2_back_speed_accuracy_plot()
# n2_back_speed_accuracy_plot_participant_type()
# plotting_dataframe = load_pickle("performance_dataframe.pickle")
# performance_nasa_tlx_3d_plot()

# main_dataframe = load_pickle("main_dataframe.pickle")
# performance_1 = main_dataframe["performance_1"]
# performance_7 = main_dataframe["performance_7"]
#
# performance_df = n_back_performance_dataframe()
# performance_speed = performance_df["performance"]
# performance_n_back = performance_df["n_back_correct"]
# # colors = ['red' if val > 20 else 'blue' for val in performance_n_back]
# nasa_tlx_ratio = [3.05, 12.6, 4.19, 6.09, 3.32, 2.17, 1.39, 3.94, 1.87, 15.33, 3.19, 2.08, 2.54, 1.2, 4.64, 3.15,
#                       3.97, 29.56, 4.69, 4.27, 3.98, 3.32]
# colors = ['red' if val > 3.6 else 'blue' for val in nasa_tlx_ratio]
#
# plt.scatter(performance_1, performance_7, c=colors, alpha=0.5, marker="o")
# x = np.linspace(5, 25, 1000)
# plt.plot(x, x, color="red")
# plt.show()