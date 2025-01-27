import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
import constants as cons

def log(message: str, verbose: bool, level="INFO"):
    """
    Logs a message with an optional timestamp and log level.
    
    Args:
        message (str): The log message.
        verbose (bool): If True, prints the log.
        level (str): The log level (e.g., INFO, ERROR). Default is "INFO".
    """
    if verbose:
        print(f"{datetime.now().isoformat()} [{level}] {message}")    


def clean_data(df):
    #1. Remove duplicates
    df = df.drop_duplicates()
    #2. Drop columns with high percentage of missing values
    df = df.drop(columns=cons.COLUMNS_TO_DROP)
    #3. Drop all rows with missing values
    df = df.dropna()

    #4. Convert DateTime into DateTime object and sort by DateTime so data is chronological:
    df[cons.DATETIME_COLUMN] = pd.to_datetime(df[cons.DATETIME_COLUMN], errors='coerce')
    if df[cons.DATETIME_COLUMN].isna().any():
        raise ValueError("Invalid DateTime entries found during preprocessing.")
    #df = df.sort_values('DateTime')

    #5. Extract hour and weekday features from DateTime column and drop the original DateTime
    df['hour'] = df[cons.DATETIME_COLUMN].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df[cons.DATETIME_COLUMN].dt.dayofweek # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
    
    # Drop raw hour column - create instead function for feature selection
    df = df.drop(columns=["hour"]) 
    
    return df

        
def split_dataset_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into features and target as DataFrames.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The DataFrame containing the features and the DataFrame containing the target.
    """
    X = df.drop(columns=cons.TARGET_COLUMN)  # Drop only the target column(s) for features
    y = df[cons.TARGET_COLUMN]  # Extract the target as a DataFrame
    X, y = X.align(y, axis=0)  # Align indices of X and y
    return X, y

def combine_Xy(X: pd.DataFrame, y : pd.DataFrame) -> pd.DataFrame:
    """Combine the features and target into a single DataFrame:
    Args:
        X (pd.DataFrame): The DataFrame containing the features.
        y (pd.DataFrame): The DataFrame containing the target.
    Returns:
        pd.DataFrame: The DataFrame containing the combined features and target.
    """
    return pd.concat([X, y], axis=1)


def align_columns(df, all_columns):
    """
    Aligns a DataFrame's columns with the union of columns from all splits.
    
    Args:
        df (pd.DataFrame): The DataFrame to align.
        all_columns (set): The union of all columns from train, val, and test.
    
    Returns:
        pd.DataFrame: The aligned DataFrame.
    """
    # Add missing columns and fill with 0
    missing_cols = all_columns - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Ensure the column order matches
    return df[list(all_columns)]

def encode_data(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, categorical_columns: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical features for train, validation, and test sets.
    
    Args:
        train (pd.DataFrame): Training set.
        val (pd.DataFrame): Validation set.
        test (pd.DataFrame): Test set.
        categorical_columns (list): List of categorical columns to encode.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Encoded train, validation, and test sets.
    """
    combined = pd.concat([train, val, test])  # Ensure consistent encoding
    combined = pd.get_dummies(combined, columns=categorical_columns, drop_first=True)
    
    # Split back into train, val, and test and Ensure consistent column alignment
    train_encoded = combined.iloc[:len(train)].reindex(columns=combined.columns)
    val_encoded =   combined.iloc[len(train):len(train) + len(val)].reindex(columns=combined.columns)
    test_encoded =  combined.iloc[len(train) + len(val):].reindex(columns=combined.columns)
    
    return train_encoded, val_encoded, test_encoded


def save_data_for_training(train, val, test, path=cons.DATA_PATH):
    """Save train, validation, and test sets to CSV files."""
    os.makedirs(path, exist_ok=True)
    train.to_csv(f'{path}/{cons.DEFAULT_TRAIN_SET_FILE}', index=False)
    val.to_csv(f'{path}/{cons.DEFAULT_VAL_SET_FILE}', index=False)
    test.to_csv(f'{path}/{cons.DEFAULT_TEST_SET_FILE}', index=False)


def load_training_data(path: str = cons.DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the train, validation, and test sets from CSV files.

    Args:
        path (str):     The path to load the CSV files. Default is 'data/'.    
        train_fn (str): The filename for the training set. Default is 'train.csv'.
        val_fn (str):   The filename for the validation set. Default is 'val.csv'.
        test_fn (str):  The filename for the test set. Default is 'test.csv'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, validation, and test sets as DataFrames.
    """
    # Load train, validation, and test sets
    train = pd.read_csv(f'{path}/{cons.DEFAULT_TRAIN_SET_FILE}')
    val = pd.read_csv(f'{path}/{cons.DEFAULT_VAL_SET_FILE}')
    test = pd.read_csv(f'{path}/{cons.DEFAULT_TEST_SET_FILE}') 
    return train, val, test