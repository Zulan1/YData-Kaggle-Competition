import pandas as pd
import numpy as np
from typing import Tuple
import sys
import os

# Add the 'app/' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), './../'))
import constants as cons

def split_dataset_Xy (df : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into features and target as DataFrames.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The DataFrame containing the features and the DataFrame containing the target.
    """
    X = df.drop(columns=cons.TARGET_COLUMN)  # Drop only the target column(s) for features
    y = df[cons.TARGET_COLUMN]  # Extract the target as a DataFrame
    return X, y

def combine_Xy (X : pd.DataFrame, y : pd.DataFrame) -> pd.DataFrame:
    """Combine the features and target into a single DataFrame:
    Args:
        X (pd.DataFrame): The DataFrame containing the features.
        y (pd.DataFrame): The DataFrame containing the target.
    Returns:
        pd.DataFrame: The DataFrame containing the combined features and target.
    """
    return pd.concat([X, y], axis=1)


def save_data_for_training(train, val, test, 
                           path=cons.DATA_PATH,
                           train_fn=cons.DEFAULT_TRAIN_SET_FILE,
                           val_fn=cons.DEFAULT_VAL_SET_FILE,
                           test_fn=cons.DEFAULT_TEST_SET_FILE):
    """Save train, validation, and test sets to CSV files."""
    train.to_csv(f'{path}/{train_fn}', index=False)
    val.to_csv(f'{path}/{val_fn}', index=False)
    test.to_csv(f'{path}/{test_fn}', index=False)

def load_training_data(path: str = cons.DATA_PATH, train_fold_fn: str = cons.DEFAULT_TRAIN_FOLD_FILE, val_fold_fn: str = cons.DEFAULT_VAL_FOLD_FILE, train_set_fn: str = cons.DEFAULT_TRAIN_SET_FILE, test_set_fn: str = cons.DEFAULT_TEST_SET_FILE) -> Tuple[list, pd.DataFrame, pd.DataFrame]:
    """Load the Cross-Validation folds and test set from CSV files:
    Args:
        path (str): The path to load the CSV files. Default is 'data/'.    
        train_fold_fn (str): The filename prefix for the training folds. Default is 'train_fold_'.
        val_fold_fn (str): The filename prefix for the validation folds. Default is 'val_fold_'.
        train_set_fn (str): The filename for the training set. Default is 'train_set.csv'.
        test_set_fn (str): The filename for the test set. Default is 'test_set.csv'.
    Returns:
        list: A list of tuples containing the training and validation folds (each one is pd.DataFrame).
        pd.DataFrame: The training set DataFrame.
        pd.DataFrame: The test set DataFrame.
    """
    folds = []
    for i in range(cons.DEFAULT_N_FOLDS):
        train_fold = pd.read_csv(f'{path}/{train_fold_fn}_{i + 1}.csv')
        val_fold = pd.read_csv(f'{path}/{val_fold_fn}_{i + 1}.csv')
        folds.append((train_fold, val_fold))
    train_set = pd.read_csv(f'{path}/{train_set_fn}')
    test_set = pd.read_csv(f'{path}/{test_set_fn}')
    
    return folds, train_set, test_set

def log(message: str, verbose: bool):
    if verbose:
        print(message)
