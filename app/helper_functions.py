import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
import constants as cons
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from transformer import DataTransformer

def transform_categorical_columns(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """Transform categorical columns using a pre-fitted OneHotEncoder.
    
    Args:
        df: Input DataFrame
        ohe: Pre-fitted OneHotEncoder
        
    Returns:
        DataFrame with transformed categorical columns
    """
    encoded_cats = ohe.transform(df[cons.CATEGORICAL])
    feature_names = ohe.get_feature_names_out(cons.CATEGORICAL)
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    df = df.drop(columns=cons.CATEGORICAL)
    return pd.concat([df, encoded_df], axis=1)

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

def drop_high_missing_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Drop columns with a high percentage of missing values."""
    return df.drop(columns=columns_to_drop, errors="ignore")

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

def save_data_for_test(df, output_path):
    X_test, y_test = split_dataset_Xy(df)
    X_test.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_FEATURES_FILE), index=False)
    y_test.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_LABELS_FILE), index=False)
    return

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

def log(message: str, verbose: bool, level: str = "INFO") -> None:
    """Log a message if verbose is True."""
    if verbose:
        print(f"{datetime.now().isoformat()} [{level}] {message}")


def one_hot_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Encode categorical columns using OneHotEncoder.
    Returns the encoded DataFrame and the encoder."""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = ohe.fit_transform(df[cons.CATEGORICAL])
    feature_names = ohe.get_feature_names_out(cons.CATEGORICAL)
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    df = df.drop(columns=cons.CATEGORICAL)  # Only drops categorical columns
    df = pd.concat([df, encoded_df], axis=1)
    return df, ohe

def get_transformer(input_path: str) -> DataTransformer:
    """Load trained transformer from pickle file."""
    transformer_path = os.path.join(input_path, cons.DEFAULT_TRANSFORMER_FILE)
    with open(transformer_path, 'rb') as f:
        return pickle.load(f)

def get_ohe(input_path: str) -> OneHotEncoder:
    """Load OneHotEncoder from file."""
    ohe_path = os.path.join(input_path, cons.DEFAULT_OHE_FILE)
    return pickle.load(open(ohe_path, 'rb'))

def get_imputer(input_path: str) -> SimpleImputer:
    """Load SimpleImputer from file."""
    imputer_path = os.path.join(input_path, cons.DEFAULT_IMPUTER_FILE)
    return pickle.load(open(imputer_path, 'rb'))

def save_imputer_to_file(imputer: SimpleImputer, path: str, verbose: bool) -> None:
    """Save the Imputer to a file."""
    with open(path, 'wb') as f:
        pickle.dump(imputer, f)
    if verbose:
        print(f"Imputer saved to {path}")

def save_ohe_to_file(ohe: OneHotEncoder, path: str, verbose: bool) -> None:
    """Save the OneHotEncoder to a file."""
    with open(path, 'wb') as f:
        pickle.dump(ohe, f)
    if verbose:
        print(f"OneHotEncoder saved to {path}")

def save_data_for_holdout(df: pd.DataFrame, output_path: str) -> None:
    """Save holdout features and labels to separate CSV files."""
    features = df.drop(columns=cons.TARGET_COLUMN)
    labels = df[cons.TARGET_COLUMN]
    
    features.to_csv(os.path.join(output_path, cons.DEFAULT_HOLDOUT_FEATURES_FILE), index=False)
    labels.to_csv(os.path.join(output_path, cons.DEFAULT_HOLDOUT_LABELS_FILE), index=False)

def save_data_for_validation(df: pd.DataFrame, output_path: str) -> None:
    """Save validation features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_VAL_SET_FILE), index=False)
    return

def save_data_for_training(df: pd.DataFrame, output_path: str) -> None:
    """Save training features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_TRAIN_SET_FILE), index=False)
    return

