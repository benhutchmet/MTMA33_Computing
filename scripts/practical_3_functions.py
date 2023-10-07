"""
Practical 3 functions
"""
from scipy.stats.stats import pearsonr, linregress
import numpy as np
import matplotlib.pyplot as plt

def read_in_data(filename):
    """
    readindata reads in numerical data into a numpy array from an ascii text file

    :param filename: the fully qualified pathname of the data file
    :return: the data from the file
    """
    input_data = np.genfromtxt(filename)
    return input_data


def extract_month(in_data, month):
    """
    Extracts a time series of data for a single month from an input array
    :param in_data: input array with four columns of data:  year, month, ccd, rain gauge measurement
    :param month: the month for which to extract data
    :return: the data for the specified month
    """

    # the monthly data is extracted by selecting every 12th row, starting from the month of interest
    # note that python arrays start at zero
    month_data = in_data[np.arange(month-1, len(in_data[:, 0]), 12), :]
    return month_data


def calibrate_data(in_data):
    """
    Takes an input file of format year, month, ccd (dependent variable, i.e. predictor),
    gauge measurement (independent variable, i.e. predictand)
    :param in_data: input file
    :return: calibration parameters
    """

    # slice the array to select the gauge and ccd data
    gauge = in_data[:, 3]
    ccd = in_data[:, 2]

    # derive a linear model using the intrinsic function linregress, imported from the scipy package
    linear_model = linregress(ccd, gauge)
    a1 = linear_model[0]
    a0 = linear_model[1]

    # return a tuple containing the calibration parameters
    return a0, a1


def make_rfe(a0, a1, ccd):
    """
    This function makes a time series of rainfall estimates, based on ccd observations and calibration parameters.
    :param a0: calibration parameter
    :param a1: calibration parameter
    :param ccd: observations
    :return: rainfall estimate timeseries
    """

    rfe = a0 + (a1 * ccd)
    return rfe

