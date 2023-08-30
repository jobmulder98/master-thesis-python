import numpy as np
import math
from clean_raw_dataset import create_clean_dataframe


def add_angle_to_dataframe(dataframe):
    dataframe["angle"] = 0
    for i in range(len(dataframe["eyesDirection"])-1):
        dataframe["angle"].iloc[i+1] = angle_between(
            dataframe["eyesDirection"].iloc[i],
            dataframe["eyesDirection"].iloc[i+1]
        )
    return dataframe


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


clean_dataframe = create_clean_dataframe()
angles_dataframe = add_angle_to_dataframe(clean_dataframe)
print(angles_dataframe.head(10))
