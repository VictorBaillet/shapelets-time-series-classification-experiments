import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import random

from utils.shapelets_transform import *
from utils.quality_measures import *
from utils.general_utils import generate_synthetic_dataset

def main(args):
    x_train, y_train, x_test, y_test = generate_synthetic_dataset(args.noise)

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