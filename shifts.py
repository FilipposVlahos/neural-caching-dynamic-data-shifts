import pandas as pd
import numpy as np

def label_shift(df):
    return df.sort_values(by='gold_hard')


def label_shift_distr(df, distribution='poisson', seed=None):
    """
    Sorts the labels in a pandas DataFrame according to a specified distribution.

    Parameters:
    - df: pandas DataFrame containing the data with labels.
    - distribution: str, the distribution to follow (default is 'poisson').
    - seed: int, seed for random number generation (default is None).

    Returns:
    - sorted_df: pandas DataFrame with labels sorted according to the specified distribution.
    """

    # Set seed for reproducibility
    np.random.seed(seed)

    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(df['label'], return_counts=True)

    # Sort labels based on specified distribution
    if distribution == 'poisson':
        # Generate labels according to Poisson distribution
        sorted_labels = np.random.poisson(size=len(df), lam=label_counts.mean())
    else:
        raise ValueError("Invalid distribution. Please choose from 'poisson'.")

    # Create a DataFrame with sorted labels
    sorted_df = df.copy()
    sorted_df['label'] = sorted_labels

    return sorted_df


def covariate_shift(data):
    return