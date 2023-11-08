import numpy as np
import pandas as pd


def mean_convergence_distance(dataframe: pd.DataFrame) -> dict:
    return np.mean(dataframe.convergenceDistance.values)

