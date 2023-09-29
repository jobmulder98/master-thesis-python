import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from src.preprocessing.HMD.clean_raw_data import create_clean_dataframe
from src.preprocessing.helper_functions.general_helpers import milliseconds_to_seconds


#  TODO: total time not looking at main task, number of fixations on other than main task,
#   mean fixation time on other object, minimum fixation time on other object, maximum fixation time on other object.


def total_time_other_object(participant_no: int, condition: int):
    dataframe = create_clean_dataframe(participant_no, condition)
    return dataframe.loc[dataframe["focusObjectTag"] == "notAssigned", "deltaSeconds"].sum()


def total_time_list(participant_no: int, condition: int):
    dataframe = create_clean_dataframe(participant_no, condition)
    return dataframe.loc[dataframe["focusObjectTag"] == "List", "deltaSeconds"].sum()


def total_time_cart(participant_no: int, condition: int):
    dataframe = create_clean_dataframe(participant_no, condition)
    return dataframe.loc[dataframe["focusObjectTag"] == "Cart", "deltaSeconds"].sum()


def total_time_main_shelf(participant_no: int, condition: int):
    dataframe = create_clean_dataframe(participant_no, condition)
    return dataframe.loc[dataframe["focusObjectTag"] == "MainShelf", "deltaSeconds"].sum()


def area_of_interest_features(participant_no: int, condition: int):
    features = {"total time other object": total_time_other_object(participant_no, condition),
                "total time main shelf": total_time_main_shelf(participant_no, condition),
                "total time list": total_time_list(participant_no, condition),
                "total time cart": total_time_cart(participant_no, condition)}
    return features


participant_number = 103
condition = 3

print("AOI features:")
for k, v in area_of_interest_features(participant_no=participant_number,
                                      condition=condition,
                                      ).items():
    print("- %s: %.2f" % (k, v))
