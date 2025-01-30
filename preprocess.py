import os
import pandas as pd
import constants as cons
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle

from app.helper_functions import split_dataset_Xy, combine_Xy, save_data_for_training, log
from app.argparser import get_preprocessing_args
from app.helper_functions import encode_data, align_columns

def encode_and_save_transformers(df: pd.DataFrame, output_path: str, verbose: bool) -> pd.DataFrame:
    """
    Perform one-hot encoding on categorical columns and save the encoder.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        output_path (str): Path to save the encoder
        verbose (bool): Whether to print verbose output
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical columns
    """
    # Initialize OneHotEncoder
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Fit and transform categorical columns
    encoded_cats = ohe.fit_transform(df[cons.CATEGORICAL])
    
    # Get feature names after encoding
    feature_names = ohe.get_feature_names_out(cons.CATEGORICAL)
    
    # Create DataFrame with encoded values
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    
    # Drop original categorical columns and join encoded ones
    df = df.drop(columns=cons.CATEGORICAL)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Save the encoder
    ohe_path = os.path.join(output_path, 'ohe.pkl')
    with open(ohe_path, 'wb') as f:
        pickle.dump(ohe, f)
    
    if verbose:
        print(f"OneHotEncoder saved to {ohe_path}")
        
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = process_datetime(df, cons.DATETIME_COLUMN)
    df = extract_time_features(df, cons.DATETIME_COLUMN)
    return df

def process_datetime(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Convert DateTime column to a proper datetime object and validate entries."""
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
    if df[datetime_column].isna().any():
        raise ValueError("Invalid DateTime entries found during preprocessing.")
    return df

def extract_time_features(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Extract hour, hour_sin, hour_cos, and day_of_week from the DateTime column."""
    df['hour'] = df[datetime_column].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df[datetime_column].dt.dayofweek
    return df.drop(columns=['hour'])


def main():
    args = get_preprocessing_args()
    full_fn = args.input_path
    log(f"Processing file: {full_fn}", args.verbose)
    df = pd.read_csv(full_fn)
    if df.empty:
        raise ValueError("The input file is empty.")
    
    ## 1. Drop unwanted columns

    df = df.drop(columns=cons.COLUMNS_TO_DROP)

    if not args.test:
        df = df.dropna()
        df = df.drop_duplicates()
        df = feature_engineering(df)
        df = encode_and_save_transformers(df, args.output_path, args.verbose)

        X, y = split_dataset_Xy(df)

        # First split: Train and Temp (for validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=cons.TRAIN_TEST_SPLIT, stratify=y, random_state=cons.RANDOM_STATE)

        # Second split: Temp -> Validation and Test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=cons.VAL_TEST_SPLIT, stratify=y_temp, random_state=cons.RANDOM_STATE)

        # Combine X and y back into DataFrames for train, validation, and test
        train = combine_Xy(X_train, y_train)
        val = combine_Xy(X_val, y_val)
        test = combine_Xy(X_test, y_test)

        # 4. Encode categorical features after splitting to avoid data leakage
        train_encoded, val_encoded, test_encoded = encode_data(
            train, val, test, categorical_columns=cons.CATEGORIAL)
        
        # 5. Column Alignment
        all_columns = set(train_encoded.columns).union(val_encoded.columns).union(test_encoded.columns)
        train_encoded = align_columns(train_encoded, all_columns)
        val_encoded = align_columns(val_encoded, all_columns)
        test_encoded = align_columns(test_encoded, all_columns)

        # 6. Multi-Level Index for Traceability
        try:
            train_encoded.set_index(cons.INDEX_COLUMNS, inplace=True, drop=True)
            val_encoded.set_index(cons.INDEX_COLUMNS, inplace=True, drop=True)
            test_encoded.set_index(cons.INDEX_COLUMNS, inplace=True, drop=True)
        except KeyError as e:
            raise KeyError(f"Ensure all index columns {cons.INDEX_COLUMNS} exist in the dataset. Missing columns: {e}")

        # 7. Align Indices of y Sets to Match X Sets
        y_train = pd.Series(y_train.values, index=train_encoded.index, name=cons.TARGET_COLUMN)
        y_val = pd.Series(y_val.values, index=val_encoded.index, name=cons.TARGET_COLUMN)
        y_test = pd.Series(y_test.values, index=test_encoded.index, name=cons.TARGET_COLUMN)

        log(f"Training set created with {len(train_encoded)} samples.", args.verbose)
        log(f"Validation set created with {len(val_encoded)} samples.", args.verbose)
        log(f"Test set created with {len(test_encoded)} samples.", args.verbose)

        # 8. Save the datasets using the helper function
        save_data_for_training(train_encoded, val_encoded, test_encoded, path=args.output_path)
    else:
        df.drop(columns=cons.TARGET_COLUMN, inplace=True)
        df.to_csv(os.path.join(args.output_path, cons.DEFAULT_TEST_SET_FILE), index=False)
        log(f"Test set saved to {cons.DEFAULT_TEST_SET_FILE}.", args.verbose)

if __name__ == '__main__':
    main()