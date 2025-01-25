import pandas as pd
from typing import Tuple
import constants as cons


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
    
    # Split back into train, val, and test
    train_encoded = combined.iloc[:len(train)]
    val_encoded = combined.iloc[len(train):len(train) + len(val)]
    test_encoded = combined.iloc[len(train) + len(val):]
    
    return train_encoded, val_encoded, test_encoded


def save_data_for_training(train, val, test, 
                           path=cons.DATA_PATH,
                           train_fn=cons.DEFAULT_TRAIN_SET_FILE,
                           val_fn=cons.DEFAULT_VAL_SET_FILE,
                           test_fn=cons.DEFAULT_TEST_SET_FILE):
    """Save train, validation, and test sets to CSV files."""
    train.to_csv(f'{path}/{train_fn}', index=False)
    val.to_csv(f'{path}/{val_fn}', index=False)
    test.to_csv(f'{path}/{test_fn}', index=False)


def load_training_data(path: str = cons.DATA_PATH, 
                       train_fn: str = cons.DEFAULT_TRAIN_SET_FILE, 
                       val_fn: str = cons.DEFAULT_VAL_SET_FILE, 
                       test_fn: str = cons.DEFAULT_TEST_SET_FILE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    train = pd.read_csv(f'{path}/{train_fn}')
    val = pd.read_csv(f'{path}/{val_fn}')
    test = pd.read_csv(f'{path}/{test_fn}')
    
    return train, val, test


def log(message: str, verbose: bool):
    if verbose:
        print(message)
