import os
import pandas as pd
from datetime import datetime
from typing import Tuple
import constants as cons
import pickle
from transformer import DataTransformer

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

def log(message: str, verbose: bool, level: str = "INFO") -> None:
    """Log a message if verbose is True."""
    if verbose:
        print(f"{datetime.now().isoformat()} [{level}] {message}")

def get_transformer(input_path: str) -> DataTransformer:
    """Load trained transformer from pickle file."""
    transformer_path = os.path.join(input_path, cons.DEFAULT_TRANSFORMER_FILE)
    with open(transformer_path, 'rb') as f:
        return pickle.load(f)

def save_data_for_test(df: pd.DataFrame, output_path: str) -> None:
    """Save holdout features and labels to separate CSV files."""
    features = df.drop(columns=cons.TARGET_COLUMN)
    labels = df[cons.TARGET_COLUMN]
    features.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_FEATURES_FILE), index=False)
    labels.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_LABELS_FILE), index=False)
    return

def save_data_for_validation(df: pd.DataFrame, output_path: str) -> None:
    """Save validation features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_VAL_SET_FILE), index=False)
    return

def save_data_for_training(df: pd.DataFrame, output_path: str) -> None:
    """Save training features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_TRAIN_SET_FILE), index=False)
    return

def get_data(input_path: str, verbose: bool):
    df = pd.read_csv(input_path)
    if verbose:
        print(f"Data loaded from {input_path}")
    if df.empty:
        raise ValueError(f"Data is empty. Check the input path: {input_path}")
    return df
