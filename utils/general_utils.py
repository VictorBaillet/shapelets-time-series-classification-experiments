"""
List and Array Manipulation Utilities

Functions:
- shuffle_lists_in_unison: Shuffles two lists in the same way, ensuring the order of elements in one list corresponds to the order in the other.
- sort_a_and_reorder_b: Sorts an array 'a' and reorders array 'b' based on the sorting order of 'a'.

Dependencies:
- numpy: Used for array operations and random number generation.
- numba: Used for just-in-time compilation to improve performance of array sorting and reordering.
- random: Used for generating random seeds.

"""

import numpy as np
import random
from numba import jit

def shuffle_lists_in_unison(l1, l2):
    """
    Shuffles two lists (l1 and l2) in the same random order.

    Args:
    l1 (list): The first list to be shuffled.
    l2 (list): The second list to be shuffled in the same order as l1.

    Returns:
    tuple: A tuple containing the shuffled versions of l1 and l2.

    Raises:
    ValueError: If the lengths of l1 and l2 are not equal.
    """
    if len(l1) != len(l2):
        raise ValueError("Lists must have the same length to be shuffled in unison.")

    seed = random.randint(0, 10000)  # Generate a random seed

    np.random.seed(seed)  # Set the random seed
    np.random.shuffle(l1)  # Shuffle the first list

    np.random.seed(seed)  # Reset the random seed to the same value
    np.random.shuffle(l2)  # Shuffle the second list
    
    return l1, l2

@jit(nopython=True)
def sort_a_and_reorder_b(a, b):
    """
    Sorts array 'a' and reorders array 'b' based on the sorting order of 'a'.

    Args:
    a (array-like): The array to be sorted.
    b (array-like): The array to be reordered according to the sorting of 'a'.

    Returns:
    tuple: A tuple containing the sorted 'a' and the reordered 'b'.
    """
    # Obtain indices that would sort 'a'
    sorted_indices = np.argsort(a)

    # Sort 'a' and reorder 'b' using these indices
    sorted_a = a[sorted_indices]
    reordered_b = b[sorted_indices]

    return sorted_a, reordered_b

def generate_synthetic_dataset(noise=1.5):
    N_train = 100 ## Number of time series in the train and test dataset
    N_test = 1000
    k = 50 ## Length of each time series 
    motif_1 = np.concatenate((np.zeros(2), 7*np.ones(14), np.zeros(2)))
    motif_2 = np.concatenate((np.zeros(2), (np.arange(8)), (6 - np.arange(7)), np.zeros(1),))
    x_train = noise*np.random.randn(N_train, k)
    x_test = noise*np.random.randn(N_test, k)
    y_train = np.zeros(N_train)
    y_test = np.zeros(N_test)
    max_motif_length = max(len(motif_1), len(motif_2))
    
    for i in range(N_train):
        idx = np.random.randint(k-max_motif_length-1)
        if i <= N_train//2:
            x_train[i][idx:idx+len(motif_1)] += motif_1
        else:
            y_train[i] = 1
            x_train[i][idx:idx+len(motif_2)] += motif_2
            
    for i in range(N_test):
        idx = np.random.randint(k-max_motif_length-1)
        if i <= N_test//2:
            x_test[i][idx:idx+len(motif_1)] += motif_1
        else:
            y_test[i] = 1
            x_test[i][idx:idx+len(motif_2)] += motif_2

    return x_train, y_train, x_test, y_test