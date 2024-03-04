"""
Shapelet Distance Calculation Module
This module provides a set of functions to calculate distances between shapelets and time series data. 

Functions:
- calculate_distance: Computes the minimum Euclidean distance between a normalized shapelet and all possible subsequences of a given time series.
- calculate_distances_for_set: Calculates distances between a single shapelet and each time series in a given set.
- dS: Calculates the distance between two shapelets.
- calculate_distance_to_cluster: Calculates the mean distance between a time series and each shapelet in a cluster.
- compute_distance_matrix: Computes a symmetric distance matrix for a given set of shapelets.

Dependencies:
- numpy: Used for numerical operations.
- numba: Used for just-in-time compilation to improve performance.
- shapelets_utils.general_utils: Contains general utility functions for shapelet operations.

Note:
The functions in this module are optimized using Numba's just-in-time compilation to enhance performance, particularly for large datasets.
"""

import numpy as np
from numba import jit
from utils.general_utils import *


@jit(nopython=True)
def calculate_distance(shapelet, time_series):
    """
    Calculates the minimum Euclidean distance between a normalized shapelet and all possible subsequences of a given time series.

    Args:
    shapelet (array-like): The shapelet to be compared.
    time_series (array-like): The time series against which the shapelet is compared.

    Returns:
    float: The minimum Euclidean distance between the shapelet and any subsequence of the time series.
    """
    shapelet = shapelet.copy() / np.linalg.norm(shapelet)
    min_distance = np.inf
    len_shapelet = len(shapelet)

    for i in range(len(time_series) - len_shapelet + 1):
        sub_series = time_series[i:i+len_shapelet]
        norm_sub_series = np.linalg.norm(sub_series)
        
        # If the norm of sub_series is 0, skip to avoid division by zero
        if norm_sub_series == 0:
            continue

        distance = 0
        for j in range(len_shapelet):
            # Incremental distance calculation
            diff = shapelet[j] - sub_series[j] / norm_sub_series
            distance += diff * diff

            # Early rejection check
            if distance >= min_distance:
                break

        if distance < min_distance:
            min_distance = distance

    return np.sqrt(min_distance)

    
@jit(nopython=True)
def calculate_distances_for_set(shapelet, x_time_series):
    """
    Calculates the distances between a single shapelet and each time series in a given set.

    Args:
    shapelet (array-like): The shapelet to compare against each time series.
    x_time_series (list or array-like): A collection of time series.

    Returns:
    numpy.ndarray: An array of distances, each representing the distance between the shapelet and a time series in the set.
    """
    distances = np.zeros(len(x_time_series))
    for i, series in enumerate(x_time_series):
        distance = calculate_distance(shapelet, series)
        distances[i] = distance
    return distances

@jit(nopython=True)
def dS(shapelet1, shapelet2):
    """
    Calculates the distance between two shapelets, ensuring the shorter shapelet is the first argument in the distance calculation.

    Args:
    shapelet1 (array-like): The first shapelet.
    shapelet2 (array-like): The second shapelet.

    Returns:
    float: The calculated distance between shapelet1 and shapelet2.
    """
    if len(shapelet1) > len(shapelet2):
        return calculate_distance(shapelet2, shapelet1)
    else:
        return calculate_distance(shapelet1, shapelet2)
    
    
def calculate_distance_to_cluster(cluster, time_series):
    """
    Calculates the mean distance between a time series and each shapelet in a cluster.

    Args:
    cluster (list or array-like): A collection of shapelets.
    time_series (array-like): A single time series.

    Returns:
    float: The mean of the distances between the time series and each shapelet in the cluster.
    """
    res = np.zeros(len(cluster))
    for i, shap in enumerate(cluster):
        res[i] = calculate_distance(np.array(shap), time_series)
    
    return np.mean(res)

def compute_distance_matrix(x_shapelet, distance_func):
    """
    Computes a symmetric distance matrix for a given set of shapelets.

    Args:
    x_shapelet (list or array-like): A collection of shapelets.
    distance_func (function): A function to compute the distance between two shapelets.

    Returns:
    numpy.ndarray: A symmetric matrix where element (i, j) represents the distance between
                   shapelets i and j as computed by distance_func.
    """
    n = len(x_shapelet)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = distance_func(x_shapelet[i], x_shapelet[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist 

    return distance_matrix
