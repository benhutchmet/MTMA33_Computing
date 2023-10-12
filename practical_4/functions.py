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