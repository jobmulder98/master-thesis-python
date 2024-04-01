import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get data directory and NASA TLX filename from environment variables
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
NASA_TLX_FILENAME = os.getenv("NASA_TLX_FILENAME")

# Construct the full path to the NASA TLX file
filename = os.path.join(DATA_DIRECTORY, "nasa_tlx", NASA_TLX_FILENAME)


def nasa_tlx_unweighted(participant, condition):
    """Retrieve unweighted NASA-TLX score for a participant and condition."""
    # Read data from Excel file
    dataframe = pd.read_excel(filename, sheet_name="nasa_tlx_unweighted", index_col=0)
    # Retrieve and return the score for the specified participant and condition
    return dataframe.at[f"p{participant}", f"c{condition}"]


def nasa_tlx_weighted(participant, condition):
    """Retrieve weighted NASA-TLX score for a participant and condition."""
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
