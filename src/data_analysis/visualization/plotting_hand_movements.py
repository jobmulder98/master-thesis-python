from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import scipy
from mpl_toolkits.mplot3d import Axes3D
import os

from src.data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.movements.filtering_head_movements import filter_head_movement_data

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low", "Mental High"]



def box_plot_hand_movements():
    # fig, ax = plt.subplots()
    # ax.set_title(f"Number of peaks for participants in all conditions".title())
    # ax.set_xlabel("Condition")
    # ax.set_xticklabels(condition_names)
    # fig.autofmt_xdate(rotation=30)
    # ax.set_ylabel("Number of peaks")
    # sns.boxplot(data=data, ax=ax, palette="Set2")
    # sns.stripplot(data=data, ax=ax, color="black", alpha=0.3, jitter=True)
    # plt.show()
    return
