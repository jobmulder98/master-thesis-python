import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from src.preprocessing.HMD.clean_raw_data import create_clean_dataframe_hmd

#  TODO: total time not looking at main task, number of fixations on other than main task,
#   mean fixation time on other object, minimum fixation time on other object, maximum fixation time on other object.


def total_time_other_object(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "notAssigned", "deltaSeconds"].sum()


def total_time_list(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "List", "deltaSeconds"].sum()


def total_time_cart(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "Cart", "deltaSeconds"].sum()


def total_time_main_shelf(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "MainShelf", "deltaSeconds"].sum()
