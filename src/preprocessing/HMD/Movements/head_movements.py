import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from src.preprocessing.HMD.clean_raw_data import create_clean_dataframe


def calculate_angular_velocity_acceleration(dataframe):
    dataframe["hmdAngularVelocity"] = dataframe["hmdEuler"].diff() / dataframe["deltaSeconds"]
    dataframe["hmdAngularVelocity"].fillna(0, inplace=True)
    dataframe['hmdAngularVelocityNorm'] = dataframe['hmdAngularVelocity'].apply(lambda acc: np.linalg.norm(acc))

    dataframe["hmdAngularAcceleration"] = dataframe["hmdAngularVelocity"].diff() / dataframe["deltaSeconds"]
    dataframe["hmdAngularAcceleration"].fillna(0, inplace=True)
    dataframe['hmdAngularAccelerationNorm'] = dataframe['hmdAngularAcceleration'].apply(lambda acc: np.linalg.norm(acc))
    dataframe['hmdAngularAccelerationNorm10'] = dataframe['hmdAngularAccelerationNorm'].rolling(10).mean()
    return


participant_number = 103
condition = 1
df = create_clean_dataframe(participant_number, condition)
calculate_angular_velocity_acceleration(df)
df["hmdAngularAccelerationNorm10"].plot()
plt.show()
print(df.head(5))
