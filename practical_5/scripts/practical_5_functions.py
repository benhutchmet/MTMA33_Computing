from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random


def readindata(file):
    '''Opens a file named by the user and puts the data into an array
    Input:  file name (full path)
    Output:  array containing the data contained in the file'''

    timeseries = np.genfromtxt(file)
    return (timeseries)


def calculate_statistics(timeseries):
    '''Calculates the statistics of rainfall from a timeseries
    Input:  One dimensional time series array
    Output: An array of length 4 containing the following: 
        [probability of rain given rain the day before, 
         probability of rain given no rain the day before, 
         mean rainfall on rainy days, 
         standard deviation rainfall amount on rainy days] '''

    # initialising the counter variables
    count_rr = 0
    count_dr = 0
    count_dd = 0
    count_rd = 0
    rainydays = timeseries[0] > 0

    # At each point in the time series, we count number of rain days followed by
    # rain (rr), or dry day (rd).
    # We also count number of dry days followed by rain (dr) or another dry day (dd).
    ndays = len(timeseries)
    for i in np.arange(1, ndays):
        if ((timeseries[i-1] > 0) & (timeseries[i] > 0)):
            count_rr = count_rr + 1

        if ((timeseries[i-1] == 0) & (timeseries[i] > 0)):
            count_dr = count_dr + 1

        if ((timeseries[i-1] == 0) & (timeseries[i] == 0)):
            count_dd = count_dd + 1

        if ((timeseries[i-1] > 0) & (timeseries[i] == 0)):
            count_rd = count_rd + 1

        if (timeseries[i] > 0):
            rainydays = np.append(timeseries[i], rainydays)

    # Based on the counts in the previous part of the code, we calculate prr (probability of rain given rain the day before)
    # and pdr (probability of rain given no rain on the day before).
    # Note that in order for this to work, we must have from __future__ import division at the top of the file.
    # otherwise, we run into problems of rounding to the nearest integer (eg 1/3 = 0)
    prr = count_rr/(count_rr + count_rd)
    pdr = count_dr/(count_dr + count_dd)
    meanrain = np.mean(rainydays)
    sdrain = np.std(rainydays)

    return ([prr, pdr, meanrain, sdrain])


def gen_timeseries(stats, length_ts):
    '''Generates a time series with user defined rainfall statistics
    Inputs:  an array of rainfall statistics in the form output by calculate_statistics 
    [probability of rain given rain the day before, 
     probability of rain given no rain the day before, 
     mean rainfall amount, 
     standard deviation rainfall amount]
    length_ts = length of time series to be generated

    Output:  an array of the generated time series'''

    # assign the elements of the array to the probabilities.
    prr = stats[0]
    pdr = stats[1]
    meanrain = stats[2]
    sdrain = stats[3]

    # initialize arrays for the generated intensity and occurrence
    gen_intensity = np.zeros(length_ts)
    gen_occurrence = np.zeros(length_ts)

    # Determine rainfall occurrence and intensity on day one.
    gen_occurrence[0] = 1
    # Commented code below is for an alternative approach where we designate
    # the first day as rainy with a 50% chance
#    if random.random() > 0.5:
#        gen_occurrence[0] = 1
#    else:
#        gen_occurrence[0] = 0

    gen_intensity[0] = random.gauss(meanrain, sdrain)

    for i in np.arange(1, length_ts):

        # Generate the intensity for each point in time.
        # Here drawing from a Gaussian distribution of intensity.
        gen_intensity[i] = np.abs(random.gauss(meanrain, sdrain))

        # generate the occurrence at each point in time

        if gen_occurrence[i-1] > 0:
            # previous day was a rain day
            if random.random() < prr:
                # set current day to rain
                gen_occurrence[i] = 1
            else:
                # set current day to dry
                gen_occurrence[i] = 0
        else:
            # previous day was dry
            if random.random() < pdr:
                # set current day to rain
                gen_occurrence[i] = 1
            else:
                # set current day to dry
                gen_occurrence[i] = 0

    outts = gen_occurrence * gen_intensity

    return (outts)


def flood_risk_timeseries(timeseries, threshold=75, accumdays=5):
    '''counts the number of times a threshold is breached and outputs the 
    risk of the threshold being breached.  
    Inputs:  time series array, rainfall threshold, days of accumulation
    Output:  Probability of a flood being in progress '''

    # initialise the flood counter array
    ndaypoints = len(timeseries)
    count_floods = np.zeros(ndaypoints)

    # loop over the time series putting changing the 0 in the counter array to 1 if the
    # threshold is breached
    for i in np.arange(accumdays, ndaypoints):  # Looping over days in time series
        # Testing whether cumulative rainfall is greater than the threshold
        if (np.sum(timeseries[i-accumdays:i]) > threshold):
            # Change the counter array element from zero to 1
            count_floods[i] = 1

    # The number of days with flood is the sum of count_floods array.
    # Calculating the probability that a day is in flood.
    probflood = np.sum(count_floods)/ndaypoints

    return (probflood)

# Define a function to assess the flood risk using the observed
# time series path and the flood threshold


def assess_flood_risk(path: str, threshold: int = 75) -> float:
    """
    Function which uses the functions readindata(), calculate_statistics(),
    gen_timeseries() and flood_risk_timeseries() to assess the flood risk, 
    according to statistics obtained from the observed time series.

    Args:
        path (str): Path to the observed time series file.
        threshold (int, optional): Flood threshold (mm). Defaults to 75.

    Returns:
        flood_risk (float): Probability of a flood being in progress.
    """

    # Read in the observed time series
    observed_ts = readindata(path)

    # Calculate the statistics of the observed time series
    stats = calculate_statistics(observed_ts)

    # Generate a synthetic time series using the statistics of the observed time series
    synthetic_ts = gen_timeseries(stats, len(observed_ts))

    # Assess the flood risk using the synthetic time series
    flood_risk = flood_risk_timeseries(synthetic_ts, threshold)

    return flood_risk

# Define a function which will assess the flood risk
# for a very long simulated time series - e.g.  1,000,000 points
# Using the functions readindata(), calculate_statistics(),
# gen_timeseries() and flood_risk_timeseries()
# Inputs to the function should be the path to the observed
# time series, the length of the time series to be generated
# and the flood threshold


def assess_flood_risk_long(path: str, length: int = 1000000, threshold: int = 75) -> float:
    """
    Function which uses the functions readindata(), calculate_statistics(),
    gen_timeseries() and flood_risk_timeseries() to assess the flood risk, 
    according to statistics obtained from the observed time series.

    Args:
        path (str): Path to the observed time series file.
        length (int, optional): Length of the synthetic time series. Defaults to 1000000.
        threshold (int, optional): Flood threshold (mm). Defaults to 75.

    Returns:
        flood_risk (float): Probability of a flood being in progress.
    """

    # Read in the observed time series
    observed_ts = readindata(path)

    # Calculate the statistics of the observed time series
    stats = calculate_statistics(observed_ts)

    # Generate a synthetic time series using the statistics of the observed time series
    synthetic_ts = gen_timeseries(stats, length)

    # Assess the flood risk using the synthetic time series
    flood_risk = flood_risk_timeseries(synthetic_ts, threshold)

    return flood_risk
