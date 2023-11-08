import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd


def total_time_other_object(dataframe: pd.DataFrame) -> float:
    condition = (dataframe["focusObjectTag"] == "notAssigned") | \
                (dataframe["focusObjectTag"] == "NPC") | \
                (dataframe["focusObjectTag"] == "Alarm")
    return dataframe.loc[condition, "deltaSeconds"].sum()


def total_time_list(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "List", "deltaSeconds"].sum()


def total_time_cart(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "Cart", "deltaSeconds"].sum()


def total_time_main_shelf(dataframe: pd.DataFrame) -> float:
    return dataframe.loc[dataframe["focusObjectTag"] == "MainShelf", "deltaSeconds"].sum()
