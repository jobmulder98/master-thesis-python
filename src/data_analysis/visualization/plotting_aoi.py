import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.eyes.fixations import add_gaze_position_to_dataframe
from src.preprocessing.hmd.eyes.area_of_interest import replace_destination_with_character, filter_location_transitions

participants = np.arange(1, 23)


def plot_fixation_location_object_tag(participant, condition, object_tags):
    if type(object_tags) == str():
        object_tags = [object_tags]
    hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
    add_gaze_position_to_dataframe(hmd_dataframe)
    hmd_dataframe["filtered_gazePosition"] = hmd_dataframe["gazePosition"]
    for index, row in hmd_dataframe.iterrows():
        if is_zero_array(row["gazePosition"]) or row["focusObjectTag"] not in object_tags:
            hmd_dataframe.at[index, "filtered_gazePosition"] = None
    hmd_dataframe_filtered = hmd_dataframe.dropna(subset=["filtered_gazePosition"])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_coords, y_coords, z_coords = zip(*hmd_dataframe_filtered["filtered_gazePosition"].apply(lambda x: tuple(x)))
    color_map = cm.ScalarMappable(cmap=cm.rainbow)
    ax.scatter(x_coords, z_coords, y_coords, c='r', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=90)


def plot_fixation_location_heat_map_object_tag(condition, object_tags):
    if isinstance(object_tags, str):
        object_tags = [object_tags]

    hmd_data_participants = []
    for participant in participants:
        hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
        add_gaze_position_to_dataframe(hmd_dataframe)
        hmd_dataframe["filtered_gazePosition"] = hmd_dataframe["gazePosition"]
        for index, row in hmd_dataframe.iterrows():
            if is_zero_array(row["gazePosition"]) or row["focusObjectTag"] not in object_tags:
                hmd_dataframe.at[index, "filtered_gazePosition"] = None
        hmd_dataframe_filtered = hmd_dataframe.dropna(subset=["filtered_gazePosition"])
        hmd_data_participants.append(hmd_dataframe_filtered)

    combined_dataframe = pd.concat(hmd_data_participants, ignore_index=True)
    x_coords, y_coords, z_coords = zip(*combined_dataframe["filtered_gazePosition"].apply(lambda x: tuple(x)))
    fig, ax = plt.subplots()
    hb = ax.hexbin(x_coords, z_coords, gridsize=100, cmap="rainbow", mincnt=1, bins='log')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    cb = fig.colorbar(hb)
    cb.set_label('Log Count')
    plt.show()


def names_of_npc_histogram(condition):
    participant_counts = {}
    for participant in participants:
        hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
        filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["NPC", "notAssigned"])
        filtered_hmd_dataframe = replace_destination_with_character(filtered_hmd_dataframe)
        npc_focus_df = filtered_hmd_dataframe[filtered_hmd_dataframe["focusObjectTag"] == "NPC"]
        object_counts = npc_focus_df["focusObjectName"].value_counts()
        participant_counts[participant] = object_counts
    combined_counts = pd.concat(participant_counts, axis=1, sort=False).sum(axis=1)
    plt.figure(figsize=(12, 8))
    combined_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Number of Times Each Object is Focused On (NPC) - Combined for All Participants')
    plt.xlabel('Object')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()


def boxplots_aoi(name_aoi):
    pass


# plot_fixation_location_object_tag(16, 3, "NPC")
names_of_npc_histogram(3)
# plot_fixation_location_heat_map_object_tag(3, "notAssigned")
