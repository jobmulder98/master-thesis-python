import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from src.preprocessing.helper_functions.general_helpers import load_pickle, write_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
participants = np.arange(1, 23)

main_dataframe = load_pickle("main_dataframe.pickle")
main_dataframe_long = load_pickle("main_dataframe_long.pickle")
column_names = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
                "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"]


def feature_exploration_3d(c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for index, row in main_dataframe.iterrows():
        ax.scatter3D(row[f"performance_{c}"], row[f"nasa_tlx_{c}"], row[f"aoi_list_{c}"])
    ax.set_xlabel(f"performance_{c}")
    ax.set_ylabel(f"nasa_tlx_{c}")
    ax.set_zlabel(f"aoi_list_{c}")
    plt.show()


# feature_exploration_3d(7)