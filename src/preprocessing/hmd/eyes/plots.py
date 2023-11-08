import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
import pandas as pd

from fixations import *


def initialize_dataframe():
    dataframe = create_clean_dataframe_hmd()
    add_gaze_position_to_dataframe(dataframe)
    add_filter_average_to_dataframe(
        dataframe,
        "gazePosition",
        "gazePositionAverage",
        3
    )
    add_degrees_per_second_to_dataframe(dataframe, "gazePositionAverage")
    return dataframe


def plot_fixations(dataframe: pd.DataFrame) -> None:
    filtered_data = dataframe["degreesPerSecond"].rolling(window=5).mean()
    plt.plot(dataframe["timeCumulative"].iloc[200:338], filtered_data[200:338])
    plt.axhline(50, color="red")
    plt.show()
    return


df = initialize_dataframe()
plot_fixations(df)
