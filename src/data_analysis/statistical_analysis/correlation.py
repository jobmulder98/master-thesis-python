import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.helper_functions.general_helpers import load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))
conditions = np.arange(1, 8)
participants = np.arange(1, 23)

main_dataframe = load_pickle("main_dataframe.pickle")
column_names = ["aoi_cart", "aoi_list", "aoi_main_shelf", "aoi_other_object", "hr", "hrv", "head_acc",
                "hand_grab_time", "hand_rmse", "nasa_tlx", "performance"]

means = []
mean_dict = {}
column_name_idx = 0
for index, column in enumerate(main_dataframe.columns):
    means.append(main_dataframe[column].mean())
    if (index + 1) % 7 == 0:
        mean_dict[column_names[column_name_idx]] = means
        means = []
        column_name_idx += 1

correlation_df = pd.DataFrame(mean_dict)

matrix = correlation_df.corr().round(2)
mask = np.triu(np.ones_like(matrix, dtype=bool))
sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag", mask=mask)
plt.xticks(rotation=30)

plt.show()
