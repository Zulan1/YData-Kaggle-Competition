import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
import constants as cons


def prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant features and performs one-hot encoding on categorical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe with raw features
        
    Returns:
        pd.DataFrame: Processed dataframe with selected and encoded features
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Select only the columns we want to use as features
    df = df[cons.FEATURES]
    
    # Get categorical columns that need encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df


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

'''
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
'''

def impute(df):
    """
    Impute missing values in the DataFrame using mode.
    
    Args:
        df (pd.DataFrame): Input DataFrame with missing values
        
    Returns:
        pd.DataFrame: DataFrame with missing values imputed
    """
    return df.fillna(df.mode().iloc[0])


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates()

def drop_high_missing_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Drop columns with a high percentage of missing values."""
    return df.drop(columns=columns_to_drop, errors="ignore")

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows with missing values."""
    return df.dropna()

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = process_datetime(df, cons.DATETIME_COLUMN)
    df = extract_time_features(df, cons.DATETIME_COLUMN)
    return df

def process_datetime(df: pd.DataFrame, datetime_column: str = cons.DATETIME_COLUMN) -> pd.DataFrame:
    """Convert DateTime column to a proper datetime object and validate entries.
    
    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column
        
    Returns:
        DataFrame with processed datetime column
    """
    if datetime_column not in df.columns:
        return df
        
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
    # Drop rows where datetime conversion failed
    return df

def extract_time_features(df: pd.DataFrame, datetime_column: str = cons.DATETIME_COLUMN) -> pd.DataFrame:
    """Extract time-based features from datetime column.
    
    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column
        
    Returns:
        DataFrame with additional time features
    """
    if datetime_column not in df.columns:
        return df
        
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        df = process_datetime(df, datetime_column)
        
    hour = df[datetime_column].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_of_week'] = df[datetime_column].dt.dayofweek
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by removing duplicates and NA values."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def split_dataset_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into features (X) and target (y)."""
    X = df.drop(columns=cons.TARGET_COLUMN)
    y = df[cons.TARGET_COLUMN]
    X, y = X.align(y, axis=0)
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


def save_data_for_training(train, val, path):
    """Save train, validation, and test sets to CSV files."""
    train.to_csv(f'{path}/{cons.DEFAULT_TRAIN_SET_FILE}', index=False)
    val.to_csv(f'{path}/{cons.DEFAULT_VAL_SET_FILE}', index=False)


# def load_training_data(path: str = cons.DATA_PATH, 
#                        train_fn: str = cons.DEFAULT_TRAIN_SET_FILE, 
#                        val_fn: str = cons.DEFAULT_VAL_SET_FILE, 
#                        holdout_fn: str = cons.DEFAULT_HOLDOUT_FILE,
#                        run_id: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Load the train, validation, and test sets from CSV files.
# 
#     Args:
#         path (str):     The path to load the CSV files. Default is 'data/'.    
#         train_fn (str): The filename for the training set. Default is 'train.csv'.
#         val_fn (str):   The filename for the validation set. Default is 'val.csv'.
#         test_fn (str):  The filename for the test set. Default is 'test.csv'.
# 
#     Returns:
#         Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, validation, and test sets as DataFrames.
#     """
#     # Load train, validation, and test sets
# 
#     if run_id is None:
#         prefix = ""
#     else:
#         prefix = f"preprocess_{run_id}/"
# 
#     train = pd.read_csv(f'{path}{prefix + cons.DEFAULT_TRAIN_SET_FILE}')
#     val = pd.read_csv(f'{path}{prefix + cons.DEFAULT_VAL_SET_FILE}')
#     holdout = pd.read_csv(f'{path}{prefix + cons.DEFAULT_HOLDOUT_FILE}') 
#     return train, val, holdout


def log(message: str, verbose: bool, level: str = "INFO") -> None:
    """Log a message if verbose is True."""
    if verbose:
        print(f"{datetime.now().isoformat()} [{level}] {message}")
