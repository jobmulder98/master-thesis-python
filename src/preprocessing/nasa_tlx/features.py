import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
NASA_TLX_FILENAME = os.getenv("NASA_TLX_FILENAME")

filename = os.path.join(DATA_DIRECTORY, "nasa_tlx", NASA_TLX_FILENAME)


def nasa_tlx_unweighted(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="nasa_tlx_unweighted", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_weighted(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="nasa_tlx_weighted")
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_features(participant, condition):
    return {"nasa-tlx weighted": nasa_tlx_weighted(participant, condition),
            "nasa-tlx unweighted": nasa_tlx_unweighted(participant, condition)}


