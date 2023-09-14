import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from src.preprocessing.HMD.clean_raw_data import create_clean_dataframe

#  TODO: total time not looking at main task, number of fixations on other than main task,
#   mean fixation time on other object, minimum fixation time on other object, maximum fixation time on other object.
#   The idea is to


def total_time_other_object():
    dataframe = create_clean_dataframe(103, 1)
