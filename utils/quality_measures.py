"""
Utility functions for statistical tests and information gain computation.

This module provides Numba-optimized functions for calculating the F-statistic, entropy, information gain,
Kruskal-Wallis statistic, and Mood's median test statistic.

Functions:
- compute_f_stat : Computes the F-statistic for a given distance array and corresponding labels.
- compute_entropy : Computes the entropy of a given label distribution.
- compute_ig : Computes the information gain (IG) for a given feature and label distribution.
- compute_kruskal_wallis_test : Computes the Kruskal-Wallis (KW) statistic for a given set of feature values and labels.
- compute_mood_median_test : Computes the Mood's median test statistic for a given set of feature values and labels.

Dependencies:
- numpy: Used for array operations and random number generation.
- tqdm: Used for displaying progress bars.
- numba: Used for just-in-time compilation to improve performance of array sorting and reordering.
- random: Used for generating random seeds.

Note:
The functions in this module are optimized using Numba's just-in-time compilation to enhance performance, particularly for large datasets.
"""

import numpy as np
from tqdm import tqdm
import random
from numba import jit

from utils.general_utils import *

@jit(nopython=True)
def compute_f_stat(x_distance, x_labels):
    """
    Computes the F-statistic for a given distance array and corresponding labels.

    Args:
    x_distance (array-like): An array of distances.
    x_labels (array-like): An array of labels corresponding to the distances.

    Returns:
    float: The computed F-statistic, representing the ratio of between-class variance to within-class variance.
    """
    C = len(np.unique(x_labels))
    n = len(x_labels)
    total_mean = np.mean(x_distance)

    # Calculate means for each class
    class_means = np.zeros(C)
    for c in range(C):
        class_distances = x_distance[x_labels == c]
        class_means[c] = np.mean(class_distances)

    # Calculate between-class variance
    between_class_variance = np.sum((class_means - total_mean) ** 2) / (C - 1)

    # Calculate within-class variance
    within_class_variance = 0.0
    for c in range(C):
        within_class_variance += np.sum((x_distance[x_labels == c] - class_means[c]) ** 2)
    within_class_variance /= (n - C)

    # Calculate F-statistic
    f_stat = between_class_variance / within_class_variance
    return f_stat

@jit(nopython=True)
def compute_entropy(x_label):
    """
    Computes the entropy of a given label distribution.

    Args:
    x_label (array-like): An array of labels.

    Returns:
    float: The entropy of the label distribution, a measure of its disorder.
    """
    C = len(np.unique(x_label))
    N = len(x_label)
    x_n = np.zeros(C)
    for label in x_label:
        x_n[int(label)] += 1
    
    return -np.sum(x_n/N*np.log(x_n/N))

@jit(nopython=True)
def xlogx(p):
    if p==0:
        return 0
    else:
        return p*np.log(p)

@jit(nopython=True)
def compute_ig(x_D, x_label):
    """
    Computes the information gain (IG) for a given feature and label distribution.

    Args:
    x_D (array-like): An array of feature values.
    x_label (array-like): An array of labels corresponding to the feature values.

    Returns:
    float: The information gain, a measure of the reduction in uncertainty in labels achieved by splitting on the feature.
    """
    # Sort x_D and reorder x_label accordingly
    x_D, x_label = sort_a_and_reorder_b(x_D, x_label)

    # C is the number of unique classes in x_label
    # N is the total number of samples
    C = len(np.unique(x_label))
    N = len(x_label)

    # Initialize a matrix to keep track of class counts for each split
    x_x_n = np.zeros((len(x_D), C))

    # Populate the matrix with cumulative counts for each class
    for k, label in enumerate(x_label):
        if k > 0:
            x_x_n[k] = x_x_n[k-1].copy()
        x_x_n[k][int(label)] += 1

    best_split_quality = np.inf

    # Iterate over possible splits
    for sp_idx in range(len(x_D)-1):
        split_quality = 0
        x_n = x_x_n[sp_idx]

        # Calculate the split quality for each class
        for i in range(len(x_n)):
            split_quality -= (sp_idx+1)*xlogx(x_n[i]/(sp_idx+1)) + (N-sp_idx-1)*xlogx((x_x_n[-1][i] - x_n[i])/(N-sp_idx-1))

        # Update the best split quality if the current one is better
        if split_quality < best_split_quality:
            best_split_quality = split_quality

    # Compute the entropy of the entire set
    H = compute_entropy(x_label)

    # Return the information gain
    return H - best_split_quality/N

@jit(nopython=True)
def compute_kruskal_wallis_test(x_D, x_label):
    """
    Computes the Kruskal-Wallis (KW) statistic for a given set of feature values and labels.

    Args:
    x_D (array-like): An array of feature values.
    x_label (array-like): An array of labels corresponding to the feature values.

    Returns:
    float: The Kruskal-Wallis statistic, a measure of the difference in central tendencies among multiple groups.
    """
    n = len(x_D)
    # Sort x_label and reorder x_D accordingly
    x_label, x_D = sort_a_and_reorder_b(x_label, x_D)
    C = len(np.unique(x_label))
    # Concatenate all samples and compute their ranks
    ranks = np.argsort(x_D) + 1  # Ranks start at 1
    
    x_class_index = np.zeros(C+1, dtype=np.int64)
    j = 0
    for i in range(len(x_label)-1):
        if x_label[i] != x_label[i+1]:
            x_class_index[j+1] = i+1
            j += 1
    x_class_index[0] = 0
    x_class_index[C] = n 
    
    # Calculate the overall mean rank
    R_bar = np.mean(ranks)

    # Calculate the KW statistic
    K = 0
    for c in range(C):
        R_i = ranks[x_class_index[c]:x_class_index[c+1]]
        K += np.sum(R_i)**2 / len(R_i)

    K *= 12 / (n * (n + 1))
    K -= 3 * (n + 1)

    return K

@jit(nopython=True)
def compute_mood_median_test(x_D, x_label):
    """
    Computes the Mood's median test statistic for a given set of feature values and labels.

    Args:
    x_D (array-like): An array of feature values.
    x_label (array-like): An array of labels corresponding to the feature values.

    Returns:
    float: The Mood's median test statistic.
    """
    median = np.median(x_D)
    C = len(np.unique(x_label))
    O = np.zeros((C, 3))
    for i in range(len(x_D)):
        label = x_label[i]
        if x_D[i] < median:
            O[int(label), 0] += 1
        else:
            O[int(label), 1] += 1
        
        O[int(label), 2] += 1
        
    mood_med = 0
    
    for c in range(C):
        e = O[c, 2] / 2
        mood_med += (O[c, 0] - e) ** 2 / e 
        mood_med += (O[c, 1] - e) ** 2 / e
        
    return mood_med