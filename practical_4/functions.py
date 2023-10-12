# Where functions are stored for practical 4
# ------------------------------------------
# Looking at the european heatwave of summer 2003

import numpy as np
import matplotlib.pyplot as plt

# Write a function called weather_generator which can
# generate and plot a time series of temperature anomalies
# Which are random in time, but with values which are
# normally distributed
def weather_generator(mean: float, std: float, n: int):
    """
    Function which generates and plots a random time series
    of temperature anomalies.

    Args:
        mean (float): The mean of the normal distribution
        std (float): The standard deviation of the normal distribution
        n (int): The number of data points to generate

    Returns:
        data (np.array): The generated data
    """

    # Generate the data
    data = np.random.normal(mean, std, n)

    # Plot the data
    plt.plot(data)
    plt.show()

    return data

# Write a function to count the anomalies
def count_anomalies(data: np.array,
                    threshold: float):
    """
    Function which counts the number of anomalies in a
    time series of temperature anomalies which exceed
    a user defined threshold.

    Args:
        data (np.array): The time series of temperature anomalies
        threshold (float): The threshold to count anomalies above

    Returns:
        count (int): The number of anomalies above the threshold
    """

    # Assert that the data is a numpy array
    assert isinstance(data, np.ndarray), "Data must be a numpy array"

    # Assert that the threshold is a float
    assert isinstance(threshold, float), "Threshold must be a float"

    # Count the number of anomalies
    count = 0

    # Loop over the data
    for i in range(len(data)):
        if data[i] > threshold:
            count += 1

    # Alternative method
    # count = len(data[data > threshold])

    return count