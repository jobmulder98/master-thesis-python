import numpy as np
import pandas as pd
from dotenv import load_dotenv
from itertools import combinations
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.preprocessing.helper_functions.general_helpers import load_pickle, write_pickle
from src.preprocessing.main.main_dataframe import participant_order

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
participants = np.arange(1, 23)

main_dataframe = load_pickle("main_dataframe.pickle")
main_dataframe_long = load_pickle("main_dataframe_long.pickle")
column_names = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv",
                "head_idle", "hand_jerk", "hand_grab_time", "nasa_tlx", "performance"]
column_dict = {"aoi_cart": "Area of Interest Cart",
               "aoi_list": "Area of Interest List",
               "aoi_main_shelf": "Area of Interest Main Shelf",
               "aoi_other_object": "Area of Interest Other Object",
               "hr": "Heart Rate",
               "hrv": "Heart Rate Variability",
               "head_idle": "Head Movement Idle",
               "hand_jerk": "Hand Smoothness (jerk)",
               "hand_grab_time": "Total Time Grabbing with Hand",
               "nasa_tlx": "NASA-TLX Score",
               "performance": "Performance"
               }
unit_dict = {"aoi_cart": "$seconds$",
               "aoi_list": "$seconds$",
               "aoi_main_shelf": "$seconds$",
               "aoi_other_object": "$seconds$",
               "hr": "$bpm$",
               "hrv": "$bpm$",
               "head_idle": "$seconds$",
               "hand_jerk": "$m/s^3$",
               "hand_grab_time": "$seconds$",
               "nasa_tlx": "$-$",
               "performance": "$seconds/item$"
               }
units = ["$seconds$", "$seconds$", "$seconds$", "$seconds$", "$bpm$", "$bpm$", "$m/s^2$", "$seconds$", "$meters$",
         "$-$", "$seconds/item$", "$m/s^3$", "$seconds$"]
condition_names = ["Baseline", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]


def feature_exploration_3d(c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for index, row in main_dataframe.iterrows():
        ax.scatter3D(row[f"performance_{c}"], row[f"nasa_tlx_{c}"], row[f"aoi_list_{c}"])
    ax.set_xlabel(f"performance_{c}")
    ax.set_ylabel(f"nasa_tlx_{c}")
    ax.set_zlabel(f"aoi_list_{c}")
    plt.show()


def covariate_scatter_plot(measure_1, measure_2):
    for i in range(1, 8):
        measure_1_values = main_dataframe[f"{measure_1}_{i}"].values
        measure_2_values = main_dataframe[f"{measure_2}_{i}"].values
        plt.scatter(measure_1_values, measure_2_values, label=f'{condition_names[i-1]}')

    plt.xlabel(column_dict[measure_1])
    plt.ylabel(column_dict[measure_2])
    plt.title(f"Covariate Scatter Plot for {column_dict[measure_1]} vs. {column_dict[measure_2]}")
    plt.legend()
    plt.savefig(f"{DATA_DIRECTORY}/images/covariate-plots/{measure_1}-vs-{measure_2}.png")
    # plt.show()


def trajectory_plot(participant, measure):
    order = participant_order(participant)
    x_values = np.arange(1, 8)
    x_ticks = []
    y_values = []
    for condition in order:
        x_ticks.append(condition_names[condition-1])
        y_values.append(main_dataframe[f"{measure}_{condition}"][participant])
    fig, ax = plt.subplots()
    plt.title(f"Trajectory plot participant {participant}, {column_dict[measure]}")
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("Condition")
    plt.xticks(x_values, labels=x_ticks)
    plt.ylabel(unit_dict[measure])
    fig.autofmt_xdate(rotation=30)
    plt.savefig(f"{DATA_DIRECTORY}/images/trajectory-plots/participant_{participant}/{measure}.png")
    # plt.show()


if __name__ == "__main__":
    # feature_exploration_3d(7)
    # for m1, m2 in combinations(column_names, 2):
    #     covariate_scatter_plot(m1, m2)
    #     plt.cla()
    #     plt.clf()
    # for participant in [4, 5, 6, 7]:
    #     for measure in column_names:
    #         trajectory_plot(participant, measure)
    pass