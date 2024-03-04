"""
Shapelet Analysis and Transformation Module
This module provides various functions for shapelet analysis and transformation in time series data.

Functions:
- generate_candidates: Generates all possible subsequences of a specified length from a time series.
- remove_self_similar: Removes self-similar shapelets from a list.
- merge: Merges two sorted arrays of shapelets and returns the top k shapelets.
- shapelet_cached_selection: Selects the top k shapelets from a time series dataset based on a quality measure.
- estimate_min_and_max: Estimates the minimum and maximum lengths for shapelets in a time series dataset.
- cluster_shapelets: Clusters shapelets into a specified number of clusters based on their similarity.
- hac_cluster_shapelets: Performs hierarchical agglomerative clustering on a set of shapelets.
- shapelets_transform: Transforms a set of time series based on their distances to a set of shapelets.
- shapelets_cluster_transform: Transforms a set of time series based on their distances to clusters of shapelets.

Dependencies:
- numpy: Used for numerical operations.
- tqdm: Used for displaying progress bars.
- sklearn.cluster: Provides AgglomerativeClustering for hierarchical clustering.
- shapelets_utils.distance_utils: Contains utilities for distance calculations.
- shapelets_utils.general_utils: Contains general utility functions for shapelet operations.
"""

import numpy as np
from tqdm import tqdm
import random
from sklearn.cluster import AgglomerativeClustering

from utils.distance_utils import *
from utils.general_utils import *

def generate_candidates(time_series, length):
    """
    Generates all possible subsequences of a specified length from a given time series.

    Args:
    time_series (list or array-like): The time series from which to generate subsequences.
    length (int): The length of each subsequence to be generated.

    Returns:
    list: A list of subsequences, each of the specified length, from the time series.

    Raises:
    ValueError: If the specified length is longer than the time series itself.
    """
    # Ensure the length is not longer than the time series
    if length > len(time_series):
        raise ValueError("Length of subsequence is longer than the time series.")

    # Generate subsequences using list comprehension
    subsequences = [[time_series[i:i + length], i] for i in range(len(time_series) - length + 1)]

    return subsequences

def remove_self_similar(shapelets):
    """
    Removes self-similar shapelets from a list, ensuring each shapelet in the returned list is distinct within the same time series.

    Args:
    shapelets (list of tuples): A list where each tuple contains a shapelet, its quality score, 
                                and its starting index in the original time series. 
                                Each tuple is in the format (shapelet, quality, index).

    Returns:
    list: A list of shapelets with self-similar shapelets removed, preserving only distinct shapelets.

    Note:
    Self-similarity is determined based on overlapping indices in the time series.
    """
    filtered_shapelets = []

    for current in shapelets:
        current_shapelet, current_quality, current_index = current
        is_similar = False

        for existing in filtered_shapelets:
            existing_shapelet, existing_quality, existing_index = existing

            # Check for overlapping indices
            max_start = max(current_index, existing_index)
            min_end = min(current_index + len(current_shapelet), existing_index + len(existing_shapelet))
            if max_start < min_end:  # Overlapping indices
                is_similar = True
                break

        if not is_similar:
            filtered_shapelets.append(current)

    return filtered_shapelets

def merge(k, k_shapelets, x_shapelet):
    """
    Merges two sorted arrays of shapelets based on their quality and returns the top k shapelets.

    Args:
    k (int): The maximum number of top shapelets to return.
    k_shapelets (numpy array): An array of tuples (shapelet, quality), sorted by quality.
    x_shapelet (numpy array): Another array of tuples (shapelet, quality), sorted by quality.

    Returns:
    numpy array: An array of the top k shapelets, sorted by quality.

    Note:
    This function assumes that both k_shapelets and x_shapelet are sorted by the quality of the shapelets.
    """
    # Initialize pointers for both arrays
    i, j = 0, 0
    merged_shapelets = []

    # Merge the two arrays
    while i < len(k_shapelets) and j < len(x_shapelet) and len(merged_shapelets) < k:
        if k_shapelets[i][1] > x_shapelet[j][1]:
            merged_shapelets.append(k_shapelets[i])
            i += 1
        else:
            merged_shapelets.append(x_shapelet[j])
            j += 1

    # Add remaining elements from k_shapelets if needed
    while i < len(k_shapelets) and len(merged_shapelets) < k:
        merged_shapelets.append(k_shapelets[i])
        i += 1

    # Add remaining elements from x_shapelet if needed
    while j < len(x_shapelet) and len(merged_shapelets) < k:
        merged_shapelets.append(x_shapelet[j])
        j += 1

    return merged_shapelets

def shapelet_cached_selection(x_T, x_labels, min_length, max_length, k, quality_test, verbose=0):
    """
    Selects the top k shapelets from a time series dataset, based on a quality measure.

    Args:
    x_T (list or array-like): The dataset of time series.
    x_labels (list or array-like): The labels associated with the time series.
    min_length (int): The minimum length of the shapelets to consider.
    max_length (int): The maximum length of the shapelets to consider.
    k (int): The number of top shapelets to select.
    quality_test (function): The function to evaluate the quality of a shapelet.
    verbose (int, optional): Verbosity level. If greater than 0, progress is shown.

    Returns:
    list: A list of the top k shapelets, each represented as a tuple (shapelet, quality, index).
    """
    k_shapelets = []
    
    # Choose iterator based on verbosity
    if verbose:
        iterator = tqdm(x_T)
    else:
        iterator = x_T

    for Ti in iterator:
        shapelets = []

        # Generate and evaluate shapelets for each length
        for l in range(min_length, max_length + 1):
            candidates = generate_candidates(Ti, l)
            for S in candidates:
                DS = calculate_distances_for_set(S[0], x_T)
                quality = quality_test(DS, x_labels)
                shapelets.append((S[0], quality, S[1]))

        shapelets.sort(key=lambda x: x[1], reverse=True)  # Sort shapelets by quality
        shapelets = remove_self_similar(shapelets)  # Remove self-similar shapelets

        k_shapelets = merge(k, k_shapelets, shapelets)  # Merge with existing top shapelets

    return k_shapelets[:k]  # Return the top k shapelets


def estimate_min_and_max(T, x_labels, quality_test):
    """
    Estimates the minimum and maximum lengths for shapelets in a time series dataset.

    Args:
    T (list or array-like): The dataset of time series.
    x_labels (list or array-like): The labels associated with the time series.
    quality_test (function): The function to evaluate the quality of a shapelet.

    Returns:
    tuple: A tuple containing the estimated minimum and maximum lengths for shapelets.

    Note:
    The estimation is based on the distribution of lengths of top shapelets in randomly selected subsets of the dataset.
    """
    shapelets = []

    for i in tqdm(range(10)):
        suffled_T, suffled_labels = shuffle_lists_in_unison(T.copy(), x_labels.copy())
        T_subset = suffled_T[:10]
        x_label_subset = suffled_labels[:10]
        current_shapelets = shapelet_cached_selection(T_subset, x_label_subset, 6, len(T[0]), 10, quality_test)
        shapelets.extend(current_shapelets)

    # Sort shapelets by their length
    shapelets.sort(key=lambda x: len(x[0]))

    # Calculate the 25th and 75th percentile lengths
    min_length = len(shapelets[int(len(shapelets) * 0.25)][0])
    max_length = len(shapelets[int(len(shapelets) * 0.75)][0])

    return min_length, max_length

    
def cluster_shapelets(S, noClusters):
    """
    Clusters shapelets into a specified number of clusters based on their similarity.

    Args:
    S (list of arrays): The list of shapelets to be clustered.
    noClusters (int): The desired number of clusters.

    Returns:
    list: A list of sets, each set representing a cluster of shapelets.

    Note:
    The clustering is performed using an average linkage strategy, where the distance between two clusters
    is defined as the average distance between all pairs of shapelets, one from each cluster.
    """
    # Convert shapelets to tuples (which are hashable) and initialize clusters
    C = [{tuple(s)} for s in S]

    while len(C) > noClusters:
        M = np.zeros((len(C), len(C)))

        # Calculate distances between all pairs of clusters
        for i, Ci in enumerate(C):
            for j, Cj in enumerate(C):
                if i != j:
                    distances = [dS(np.array(cl), np.array(ck)) for cl in Ci for ck in Cj]
                    M[i, j] = sum(distances) / len(distances)  # Average linkage

        # Find the pair of clusters with the smallest distance
        best = np.inf
        x, y = 0, 0
        for i in range(len(C)):
            for j in range(len(C)):
                if M[i, j] < best and i != j:
                    x, y = i, j
                    best = M[i, j]

        # Merge the closest pair of clusters
        C_new = C[x].union(C[y])
        C = [C[k] for k in range(len(C)) if k != x and k != y]
        C.append(C_new)

    return C

def hac_cluster_shapelets(S, noClusters):
    """
    Performs hierarchical agglomerative clustering (HAC) on a set of shapelets.

    Args:
    S (list of arrays): The list of shapelets to be clustered.
    noClusters (int): The desired number of clusters.

    Returns:
    list: A list of sets, each set representing a cluster of shapelets.
    """
    # Compute the distance matrix
    distance_matrix = compute_distance_matrix(S, dS)

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=noClusters, metric='precomputed', linkage='complete')
    labels = clustering.fit_predict(distance_matrix)

    # Convert shapelets to tuples and initialize clusters
    shapelets_tuples = [tuple(s) for s in S]
    C = [set() for _ in range(noClusters)]

    # Group shapelets by their cluster labels
    for shapelet, label in zip(shapelets_tuples, labels):
        C[label].add(shapelet)

    return C

def shapelets_transform(x_shapelet, x_T):
    """
    Transforms a set of time series based on their distances to a set of shapelets.

    Args:
    x_shapelet (list or array-like): A collection of shapelets.
    x_T (list or array-like): A collection of time series.

    Returns:
    numpy.ndarray: A 2D array where each element (i, j) represents the distance from the i-th time series in x_T to the j-th shapelet in x_shapelet.
    """
    res = np.zeros((len(x_T), len(x_shapelet)))
    for i,T in enumerate(x_T):
        for j, shap in enumerate(x_shapelet):
            res[i, j] = calculate_distance(shap, T)
                   
    return res

def shapelets_cluster_transform(clusters, x_T):
    """
    Transforms a set of time series based on their distances to clusters of shapelets.

    Args:
    clusters (list of sets): A collection of clusters, each containing shapelets.
    x_T (list or array-like): A collection of time series.

    Returns:
    numpy.ndarray: A 2D array where each element (i, j) represents the distance from the i-th time series in x_T to the j-th cluster of shapelets in clusters.
    """
    res = np.zeros((len(x_T), len(clusters)))
    for i,T in enumerate(x_T):
        for j, c in enumerate(clusters):
            res[i, j] = calculate_distance_to_cluster(c, T)
                   
    return res
