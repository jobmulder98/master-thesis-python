import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from src.preprocessing.helper_functions.general_helpers import is_zero_array, load_pickle, write_pickle
from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.hmd.eyes.fixations import add_gaze_position_to_dataframe
from src.preprocessing.hmd.eyes.area_of_interest import replace_destination_with_character, filter_location_transitions

participants = np.arange(1, 23)
conditions = np.arange(1, 8)


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
    plt.show()


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

    # main_shelf = patches.Rectangle((-0.15, 1.1), 3.05, 0.6, linewidth=1, edgecolor='black', facecolor='none')
    # pilot_woman = patches.Circle((4, 0), 0.3, linewidth=1, edgecolor='black', facecolor='none')
    # viking_woman = patches.Circle((-0.8, 1), 0.3, linewidth=1, edgecolor='black', facecolor='none')
    # large_boy = patches.Circle((-0.4, 0.7), 0.3, linewidth=1, edgecolor='black', facecolor='none')
    # dancing_character = patches.Circle((-0.6, 0), 0.3, linewidth=1, edgecolor='black', facecolor='none')
    # judge_man = patches.Circle((2.7, 2), 0.3, linewidth=1, edgecolor='black', facecolor='none')
    # punk_man = patches.Circle((-0.8, 4.2), 0.3, linewidth=1, edgecolor='black', facecolor='none')
    # business_man = patches.Circle((-7.7, -0.35), 0.3, linewidth=1, edgecolor='black', facecolor='none')
    # objects_list = [main_shelf, pilot_woman, viking_woman, large_boy, dancing_character, judge_man, punk_man, business_man]
    # for object_patch in objects_list:
    #     ax.add_patch(object_patch)

    plt.show()


def ray_direction_histogram(condition, object_tags):
    if isinstance(object_tags, str):
        object_tags = [object_tags]

    ray_direction_data = []
    for participant in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
        filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["notAssigned", "NPC"], 0.1)
        window_dataframe = filtered_hmd_dataframe[filtered_hmd_dataframe["focusObjectTag"].isin(object_tags)]
        window_dataframe["rayDirection"] = window_dataframe["rayDirection"].apply(
            lambda x: x if not is_zero_array(x) else None)
        hmd_dataframe_filtered = window_dataframe.dropna(subset=["rayDirection"])
        ray_direction_data.append(hmd_dataframe_filtered)

    combined_dataframe = pd.concat(ray_direction_data, ignore_index=True)
    x_coords, y_coords, z_coords = zip(*combined_dataframe["rayDirection"])
    xz_angle_rad = np.arctan2(z_coords, x_coords)
    xz_angle_degrees = np.degrees(xz_angle_rad)
    xz_angle_degrees = (xz_angle_degrees + 180) % 360 - 180

    fig, ax = plt.subplots()
    ax.hist(xz_angle_degrees, bins=360, color="skyblue", edgecolor="black")
    ax.set_title(f"Number of rays in degrees for condition {condition}")
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Total rays')
    plt.show()


def ray_origin_plot(condition, object_tags):
    if isinstance(object_tags, str):
        object_tags = [object_tags]

    ray_direction_data = []
    for participant in participants:
        hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
        filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["notAssigned", "NPC"], 0.1)
        window_dataframe = filtered_hmd_dataframe[filtered_hmd_dataframe["focusObjectTag"].isin(object_tags)]
        window_dataframe["rayOrigin"] = window_dataframe["rayOrigin"].apply(
            lambda x: x if not is_zero_array(x) else None)
        hmd_dataframe_filtered = window_dataframe.dropna(subset=["rayOrigin"])
        ray_direction_data.append(hmd_dataframe_filtered)

    combined_dataframe = pd.concat(ray_direction_data, ignore_index=True)
    x_coords, y_coords, z_coords = zip(*combined_dataframe["rayOrigin"])

    fig, ax = plt.subplots()
    ax.scatter(x_coords, z_coords, color="black")
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.show()


def barplot_names_of_npc(condition):
    participant_counts = {}
    for participant in participants:
        hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
        filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["NPC", "notAssigned"])
        filtered_hmd_dataframe = replace_destination_with_character(filtered_hmd_dataframe)
        npc_focus_df = filtered_hmd_dataframe[filtered_hmd_dataframe["focusObjectTag"] == "NPC"]
        object_counts = npc_focus_df["focusObjectName"].value_counts()
        participant_counts[participant] = object_counts
    combined_counts = pd.concat(participant_counts, axis=1, sort=False).sum(axis=1)
    combined_counts["Other"] = combined_counts[combined_counts < 50].sum()
    combined_counts = combined_counts[combined_counts > 50]
    plt.figure(figsize=(12, 8))
    combined_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Number of Times Each NPC is Focused On for All Participants')
    plt.xlabel('Object')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def boxplots_aoi(name_aoi):
    """
        The name_aoi parameter is one of the names of the aois in the pickle file.
        The options of this name are ["list", "cart", "main_shelf", "other_object", "invalid", "transition"]
    """
    aoi_dictionary = load_pickle("aoi_results.pickle")
    plot_dictionary = {}
    for condition in conditions:
        aoi_values = []
        for participant in participants:
            aoi_values.append(aoi_dictionary[condition][participant-1][name_aoi])
        plot_dictionary[condition] = aoi_values
    # fig, ax = plt.subplots()
    # plot_title = f"Total time looking at {name_aoi} for each condition".replace("_", " ").title()
    # ax.set_title(plot_title)
    # ax.set_xlabel("Condition")
    # ax.set_ylabel("Time (s)")
    # ax.boxplot(plot_dictionary.values())
    # ax.set_xticklabels(plot_dictionary.keys())
    # plt.show()
    return plot_dictionary


def barplot_total_times_condition(condition, name_aoi):
    """
        The name_aoi parameter is one of the names of the aois in the pickle file.
        The options of this name are ["list", "cart", "main_shelf", "other_object", "invalid", "transition"]
    """
    aoi_dictionary = load_pickle("aoi_results.pickle")
    participant_dictionary = {}
    for participant in participants:
        participant_dictionary[participant] = aoi_dictionary[condition][participant-1][name_aoi]
    plot_title = (f"Total time for each subject looking at {name_aoi} for condition {condition}"
                  .replace("_", " ").title())
    plt.title(plot_title)
    plt.bar(list(participant_dictionary.keys()), participant_dictionary.values(), color="skyblue", edgecolor="black")
    plt.xlabel("Participants")
    plt.xticks(list(participant_dictionary.keys()))
    plt.ylabel("Time (s)")
    plt.show()


def barplot_total_time():
    aoi_dictionary = load_pickle("aoi_results.pickle")
    totals = {}
    for condition, participants in aoi_dictionary.items():
        condition_total = {"list": 0, "cart": 0, "main_shelf": 0, "other_object": 0, "invalid": 0, "transition": 0}
        for participant in participants:
            for key in condition_total.keys():
                condition_total[key] += participant[key]
        totals[condition] = condition_total
    categories = list(condition_total.keys())
    conditions = list(aoi_dictionary.keys())
    values = np.array([[total[key] / sum(total.values()) * 100 for key in categories] for total in totals.values()])
    fig, ax = plt.subplots()
    bottom = np.zeros(len(conditions))
    for i, category in enumerate(categories):
        ax.bar(conditions, values[:, i], label=category, bottom=bottom, width=0.5)
        bottom += values[:, i]
    ax.set_xlabel('Conditions')
    ax.set_ylabel('Percentage of Time Spent')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def barplot_other_object_over_time(condition: int, names_aoi, window_size: int) -> dict:
    """
        The name_aoi parameter is one of the names of the aois in the pickle file.
        The options of this name are ["list", "cart", "main_shelf", "other_object", "invalid", "transition"]
    """
    if isinstance(names_aoi, str):
        names_aoi = [names_aoi]

    time_window_dict = {}
    for participant in participants:
        hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
        filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["notAssigned", "NPC"], 0.1)
        df = replace_destination_with_character(filtered_hmd_dataframe)
        for start_time in range(0, 120, window_size):
            end_time = start_time + window_size if start_time != 110 else 122
            window_name = f"{start_time}-{end_time}"
            window_df = df[(df["timeCumulative"] >= start_time) & (df["timeCumulative"] < end_time)]
            total_time = window_df[window_df["focusObjectTag"].isin(names_aoi)]["deltaSeconds"].sum()
            time_window_dict.setdefault(window_name, 0)
            time_window_dict[window_name] += total_time
    for key in time_window_dict:
        time_window_dict[key] /= len(participants)
    plot_title = f"Average time for each subject looking at another object for condition {condition} over time"
    plt.title(plot_title)
    plt.bar(list(time_window_dict.keys()), time_window_dict.values(), color="skyblue", edgecolor="black")
    plt.xlabel("Time window (s)")
    plt.xticks(list(time_window_dict.keys()))
    plt.ylabel("Total time (s)")
    plt.xticks(rotation=45, ha='right')
    plt.show()


def lineplot_aoi_over_time(names_aoi, window_size: int):
    if isinstance(names_aoi, str):
        names_aoi = [names_aoi]

    time_window_dict = {condition: {} for condition in conditions}
    for condition in conditions:
        for participant in participants:
            hmd_dataframe = create_clean_dataframe_hmd(participant, condition)
            filtered_hmd_dataframe = filter_location_transitions(hmd_dataframe, ["notAssigned", "NPC"], 0.1)
            df = replace_destination_with_character(filtered_hmd_dataframe)
            for start_time in range(2, 122, window_size):
                end_time = start_time + window_size
                window_name = f"{start_time-2}-{end_time-2}"
                window_df = df[(df["timeCumulative"] >= start_time) & (df["timeCumulative"] < end_time)]
                total_time = window_df[window_df["focusObjectTag"].isin(names_aoi)]["deltaSeconds"].sum()
                time_window_dict[condition].setdefault(window_name, []).append(total_time)
    average_time_dict = {condition: {key: np.mean(values) for key, values in time_windows.items()} for
                         condition, time_windows in time_window_dict.items()}
    plt.figure(figsize=(10, 5))
    for condition in conditions:
        plt.plot(list(average_time_dict[condition].keys()), list(average_time_dict[condition].values()), marker='o',
                 label=f'Condition {condition}')

    if isinstance(names_aoi, list):
        if len(names_aoi) > 1:
            names_aoi_str = f"{', '.join(names_aoi[:-1])} and {names_aoi[-1]}"
        else:
            names_aoi_str = names_aoi[0]
    else:
        names_aoi_str = names_aoi
    plot_title = f"Average Time Looking at {names_aoi_str} over time for all conditions".replace("_", " ").title()
    plt.title(plot_title)
    plt.xlabel("Time window (s)")
    plt.ylabel("Average time (s)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()


# lineplot_aoi_over_time("List", 10)
# barplot_other_object_over_time(3, ["notAssigned", "NPC"], 10)
# plot_fixation_location_object_tag(16, 3, "NPC")
# barplot_names_of_npc(3)
# boxplots_aoi("invalid")
# boxplots_aoi("transition")
# barplot_total_times_condition(5, "list")
# barplot_total_time()

# plot_fixation_location_heat_map_object_tag(6, ["NPC", "notAssigned"])
# ray_direction_histogram(4, ["NPC", "notAssigned"])
# ray_origin_plot(3, ["NPC", "notAssigned"])
