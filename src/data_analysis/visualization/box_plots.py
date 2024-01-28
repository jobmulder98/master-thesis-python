import numpy as np
import pandas as pd
from dotenv import load_dotenv
from itertools import combinations
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.preprocessing.helper_functions.general_helpers import load_pickle, write_pickle
from src.data_analysis.visualization.plotting_aoi import boxplots_aoi
from src.data_analysis.visualization.plotting_ecg import heart_rate_boxplot, heart_rate_variability_boxplot
from src.data_analysis.visualization.plotting_general import box_plot_general
from src.data_analysis.visualization.plotting_head_movements import box_plot_idle_time
from src.data_analysis.visualization.plotting_hand_movements import box_plot_hand_movements_grab_time, box_plot_jerk
from src.data_analysis.visualization.plotting_hand_movements import box_plot_hand_movements_grab_time, box_plot_jerk
from src.data_analysis.statistical_analysis.behavior_features import (
    box_plot_percentage_list_isgrabbing, ratio_time_list_items, ratio_frequency_list_items
)
from src.data_analysis.helper_functions.visualization_helpers import save_figure

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


def save_all_box_plots():
    # aois = ["list", "cart", "main_shelf", "other_object"]
    # for aoi in aois:
    #     boxplots_aoi(aoi)
    #     save_figure(f"boxplot-aoi-{aoi}.png")
    #
    # box_plot_hand_movements_grab_time()
    # save_figure("boxplot-hand-grab-time.png")
    #
    # box_plot_jerk()
    # save_figure(f"boxplot-hand-jerk.png")
    #
    # box_plot_general("nasa-tlx weighted")
    # save_figure("boxplot-nasa-tlx.png")
    #
    # box_plot_general("seconds/item window")
    # save_figure("boxplot-performance.png")
    #
    # heart_rate_boxplot("ecg_data_unfiltered.pickle", np.arange(1, 22), conditions)
    # save_figure("boxplot-heart-rate.png")
    #
    # heart_rate_variability_boxplot("ecg_data_unfiltered.pickle", np.arange(1, 22), conditions)
    # save_figure("boxplot-heart-rate-variability.png")
    #
    # box_plot_idle_time()
    # save_figure("boxplot-head-idle.png")

    ratio_frequency_list_items()
    save_figure("boxplot-behavior-ratio-frequency.png")

    ratio_time_list_items()
    save_figure("boxplot-behavior-ratio-time.png")

    box_plot_percentage_list_isgrabbing()
    save_figure("boxplot-behavior-grab.png")


save_all_box_plots()
