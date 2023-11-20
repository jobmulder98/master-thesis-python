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
    dataframe = pd.read_excel(filename, sheet_name="nasa_tlx_weighted", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_mental(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="mental", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_physical(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="physical", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_temporal(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="temporal", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_performance(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="performance", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_effort(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="effort", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_frustration(participant, condition):
    dataframe = pd.read_excel(filename, sheet_name="frustration", index_col=0)
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_features(participant, condition):
    return {"nasa-tlx weighted": nasa_tlx_weighted(participant, condition),
            "nasa-tlx unweighted": nasa_tlx_unweighted(participant, condition),
            "nasa-tlx mental": nasa_tlx_mental(participant, condition),
            "nasa-tlx physical": nasa_tlx_physical(participant, condition),
            "nasa-tlx temporal": nasa_tlx_temporal(participant, condition),
            "nasa-tlx performance": nasa_tlx_performance(participant, condition),
            "nasa-tlx effort": nasa_tlx_effort(participant, condition),
            "nasa-tlx frustration": nasa_tlx_frustration(participant, condition)}
