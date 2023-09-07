import numpy as np
import pandas as pd


def interpolate_zero_arrays(df, column_name):
    """
    Interpolate over zero arrays in a given column of a pandas DataFrame.

    Parameters:
    - df: The DataFrame containing the data.
    - column_name: The name of the column with arrays of 3D coordinates.

    Returns:
    - df: The DataFrame with interpolated values for zero arrays.
    """

    # Define a function to check if an array is [0, 0, 0]
    def is_zero_array(arr):
        return np.array_equal(arr, np.array([0, 0, 0]))

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Create a mask to identify rows with zero arrays in the specified column
    mask = df[column_name].apply(is_zero_array)

    # Iterate over each coordinate (x, y, z) and interpolate separately
    for i in range(3):
        col_name = f"{column_name}_{i}"

        # Extract the i-th coordinate from the arrays
        df[col_name] = df[column_name].apply(lambda arr: arr[i])

        # Replace the zero values with NaN for interpolation
        df.loc[mask, col_name] = np.nan

        # Interpolate over NaN values using the 'linear' method
        df[col_name] = df[col_name].interpolate(method='linear')

    # Drop the original column with arrays of coordinates
    df.drop(columns=[column_name], inplace=True)

    # Create a new column with interpolated arrays
    df[column_name] = df.apply(lambda row: np.array([row[f"{column_name}_{i}"] for i in range(3)]), axis=1)

    # Drop the temporary coordinate columns
    df.drop(columns=[f"{column_name}_{i}" for i in range(3)], inplace=True)

    return df


# Example usage:
data = {'Coordinates': [np.array([1, 2, 3]), np.array([0, 0, 0]), np.array([4, 5, 6]), np.array([0, 0, 0]),
                        np.array([7, 8, 9])]}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

df = interpolate_zero_arrays(df, 'Coordinates')

print("\nDataFrame after interpolation:")
print(df)