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

def one_hot_encode(df):
    """
    Encode categorical columns using OneHotEncoder
    return the encoded DataFrame and the encoder
    """
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform categorical columns
    encoded_cats = ohe.fit_transform(df[cons.CATEGORICAL])
    
    # Get feature names after encoding
    feature_names = ohe.get_feature_names_out(cons.CATEGORICAL)
    
    # Create DataFrame with encoded values
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    
    # Drop original categorical columns and join encoded ones
    df = df.drop(columns=cons.CATEGORICAL)
    df = pd.concat([df, encoded_df], axis=1)

    return df, ohe

def save_ohe_to_file(ohe, path, verbose):
    """
    Save the OneHotEncoder to a file
    """
    with open(path, 'wb') as f:
        pickle.dump(ohe, f)
    if verbose:
        print(f"OneHotEncoder saved to {path}")
    

def save_data_for_prediction(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
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
    return df.drop(columns=['hour', 'DateTime'])

def preprocess_towards_training(df):
    df = df.drop(columns=cons.COLUMNS_TO_DROP)
    df = feature_engineering(df)
    df, ohe = one_hot_encode(df)
    df = df.drop(columns=cons.INDEX_COLUMNS)
    return df, ohe


def main():
    args = get_preprocessing_args()
    run_id = args.run_id
    output_path = os.path.join(args.output_path, f"preprocess_{run_id}")
    os.makedirs(output_path, exist_ok=True)
    train_mode = not args.test
    test_mode = args.test
    full_fn = args.input_path
    log(f"Processing file: {full_fn}", args.verbose)
    df = pd.read_csv(full_fn)
    if df.empty:
        raise ValueError("The input file is empty.")

    if train_mode:
        df = df.drop_duplicates()
        df = df.dropna(subset=[col for col in df.columns if col not in cons.COLUMNS_TO_DROP])
        X, y = split_dataset_Xy(df)
        X_internal, X_holdout, y_internal, y_holdout = train_test_split(
            X, y, test_size=cons.TRAIN_TEST_SPLIT, stratify=y, random_state=cons.RANDOM_STATE)
        
        holdout_labels = pd.DataFrame(y_holdout, columns=[cons.TARGET_COLUMN]) 
        labels_path = os.path.join(output_path, cons.DEFAULT_LABELS_FILE)
        holdout_labels.to_csv(labels_path, index=False)
        holdout_data = pd.concat([X_holdout, y_holdout], axis=1)
        internal_data = pd.concat([X_internal, y_internal], axis=1)
        processed_internal_data, ohe = preprocess_towards_training(internal_data)
        save_ohe_to_file(ohe, f"{output_path}/ohe.pkl", args.verbose)

        X_train, X_val, y_train, y_val = train_test_split(
            processed_internal_data, y_internal, test_size=cons.VAL_TEST_SPLIT, stratify=y_internal, random_state=cons.RANDOM_STATE)
        
        train = pd.concat([X_train, y_train], axis=1)
        val = pd.concat([X_val, y_val], axis=1)
        save_data_for_training(train, val, output_path)
        if args.verbose:
            print(f"Saved preprocessed data to {output_path}")
        X_holdout.to_csv(os.path.join(output_path, cons.DEFAULT_HOLDOUT_FILE), index=False)
    
    if test_mode:
        df.drop(columns=cons.COLUMNS_TO_DROP, inplace=True)
        df.drop(columns=cons.INDEX_COLUMNS, inplace=True)
        output_path = os.path.join(output_path, cons.DEFAULT_PROCESSED_TEST_FILE)
        df.to_csv(output_path, index=False)
        log(f"Test set saved to {output_path}.", args.verbose)

    




        
        
        
        



    

    
    
    



if __name__ == '__main__':
    main()