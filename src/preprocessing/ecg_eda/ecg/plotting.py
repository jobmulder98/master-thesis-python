import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
ECG_SAMPLE_RATE = int(os.getenv("ECG_SAMPLE_RATE"))


def plot_heart_rate(corrected_peaks: list):
    fig, ax = plt.subplots()
    for index, peaks in enumerate(corrected_peaks):
        rr_intervals = np.diff(peaks)
        heart_rate = 60 / (rr_intervals / ECG_SAMPLE_RATE)
        time_axis = np.cumsum(rr_intervals) / ECG_SAMPLE_RATE
        ax.plot(time_axis, heart_rate, marker="o", label=index+1)
    ax.set_title("Heart Rate Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.legend()
    plt.show()


