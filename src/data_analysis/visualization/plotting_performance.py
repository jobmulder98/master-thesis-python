from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

from src.data_analysis.helper_functions.visualization_helpers import increase_opacity_condition
from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle, pickle_exists
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.performance.filtering import count_errors, seconds_per_item, product_lists

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
participants = np.arange(1, 23)
conditions = np.arange(1, 8)
condition_names = ["No Stimuli", "Visual Low", "Visual High", "Auditory Low", "Auditory High", "Mental Low",
                   "Mental High"]


def speed_accuracy_plot():
    if pickle_exists("performance_dataframe.pickle"):
        plotting_dataframe = load_pickle("performance_dataframe.pickle")
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
        plotting_dataframe["condition"] + np.random.uniform(-0.05, 0.05, len(plotting_dataframe)),
        plotting_dataframe["performance"],
        marker="o",
        color=plotting_dataframe["color"],
        alpha=0.75,
    )
    ax.set_title(f"The seconds/item vs errors for all participants in all condition".title())
    ax.set_xlabel("Errors")
    ax.set_xticks(conditions)
    ax.set_xticklabels(condition_names)
    ax.set_ylabel("Seconds/Item")
    ax.legend(handles=legend_elements, title='Errors', loc='upper left')
    plt.show()


speed_accuracy_plot()
# plotting_dataframe = load_pickle("performance_dataframe.pickle")
# print(plotting_dataframe)