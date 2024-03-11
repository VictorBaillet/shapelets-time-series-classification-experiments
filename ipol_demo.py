import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils.shapelets_transform import *
from utils.quality_measures import *
from utils.general_utils import generate_synthetic_dataset

def main(args):
    x_train, y_train, x_test, y_test = generate_synthetic_dataset(args.noise)

    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    # Plot the first time series from X_train
    axes2[0].plot(x_train[0])
    axes2[0].set_title("First class time series example")
    
    # Plot the last time series from X_train
    axes2[1].plot(x_train[-1])
    axes2[1].set_title("Second class time series example")

    # Save the new figure to a different file
    plt.savefig("time_series_examples.png")

    x_shapelet = shapelet_cached_selection(x_train, y_train, args.shapelet_lenght_min, 
                                           args.shapelet_lenght_max, args.n_shapelet, compute_f_stat, verbose=1)

    shapelet_idx = [0, 1, 70, 71]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    for i, idx in enumerate(shapelet_idx):
        axes[i//2][i%2].plot(x_shapelet[idx][0])
        axes[i//2][i%2].set_title(f"Shapelet number {idx}")

    plt.savefig("results.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Shapelet classification of time series'))
    parser.add_argument('--noise', type=float,)
    parser.add_argument('--shapelet_lenght_min', type=int, default=20,)
    parser.add_argument('--shapelet_lenght_max', type=int, default=30,)
    parser.add_argument('--n_shapelet', type=int, default=100,)
    args = parser.parse_args()

    main(args)
