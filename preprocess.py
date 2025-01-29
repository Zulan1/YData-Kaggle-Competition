import pandas as pd
import constants as cons
import os
import pickle

from sklearn.model_selection import train_test_split
from app.helper_functions import align_columns, clean_data, split_dataset_Xy, combine_Xy, save_data_for_training, log, encode_data
from app.argparser import get_preprocessing_args
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

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
    return df.drop(columns=['hour', cons.DATETIME_COLUMN])

def one_hot_encode_test(df, encoder_path=None, verbose=False):
    """One hot encode categorical columns using a pre-fitted encoder.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        encoder_path (str): Path to the saved encoder
        verbose (bool): Whether to print verbose output
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical columns
    """
    if encoder_path is None:
        raise ValueError("encoder_path must be provided for test data encoding")
        
    with open(encoder_path, 'rb') as f:
        one_hot = pickle.load(f)
        
    # Transform categorical columns using loaded encoder
    categorical_data = df[cons.CATEGORICAL]
    encoded_data = one_hot.transform(categorical_data)
    
    # Convert encoded array back to dataframe with proper column names
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot.get_feature_names_out(cons.CATEGORICAL))
    
    # Drop original categorical columns and add encoded ones
    df = df.drop(columns=cons.CATEGORICAL)
    df = pd.concat([df, encoded_df], axis=1)

    return df

def one_hot_encode_train(df, encoder_path=None, verbose=False):
    """One hot encode categorical columns and optionally save the fitted encoder.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        encoder_path (str, optional): Path to save the fitted encoder
        verbose (bool): Whether to print verbose output
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical columns
    """
    # One hot encode categorical columns
    one_hot = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_data = df[cons.CATEGORICAL]  # Fixed typo in constant name
    encoded_data = one_hot.fit_transform(categorical_data)
    
    # Convert encoded array back to dataframe with proper column names
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot.get_feature_names_out(cons.CATEGORICAL))
    
    # Drop original categorical columns and add encoded ones
    df = df.drop(columns=cons.CATEGORICAL)
    df = pd.concat([df, encoded_df], axis=1)

    # Save the fitted encoder if path provided
    if encoder_path:
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        with open(encoder_path, 'wb') as f:
            pickle.dump(one_hot, f)
        log(f"Saved one-hot encoder to {encoder_path}", verbose)
        
    return df


def main():
    args = get_preprocessing_args()

    # 1. Validate and load CSV file

    full_fn = args.input_path
    if not os.path.exists(full_fn):
        raise FileNotFoundError(f"Input file '{full_fn}' does not exist.")
    
    df = pd.read_csv(full_fn)
    if df.empty:
        raise ValueError("Input dataset is empty.")
    log(f"Processing file: {full_fn}", args.verbose)

    ## 1. Drop the columns that won't be used -- for either training or prediction

    df = df.drop(columns=cons.COLUMNS_TO_DROP)
    df = df.drop(columns=cons.INDEX_COLUMNS)


    if not args.test:
        df = one_hot_encode_train(df, args.one_hot_encoder_path, args.verbose)
    else:
        df = one_hot_encode_test(df, args.one_hot_encoder_path, args.verbose)

    ## 3. Feature Engineering

    df = process_datetime(df, cons.DATETIME_COLUMN)
    df = extract_time_features(df, cons.DATETIME_COLUMN)

    ## 4. Impute missing values
    if not args.test:
        imputer = SimpleImputer(strategy='mean')
        df = imputer.fit(df)
        # Save the fitted imputer
        os.makedirs(os.path.dirname(args.imputer_path), exist_ok=True)
        with open(args.imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        log(f"Saved imputer to {args.imputer_path}", args.verbose)

        df = df.dropna()
        df = df.drop_duplicates()
    
    ## 5. Save the processed data


    if not args.test: 
        # 3. Split the data into features (X) and target (y)
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
        save_data_for_training(train_encoded, val_encoded, test_encoded, path=args.out_path)

    else:
        # 9. Process the external test set (for prediction)
        df.to_csv(os.path.join(args.csv_path, cons.DEFAULT_EXTERNAL_RAW_TEST_FILE), index=False)
        log(f"Test set saved to {cons.DEFAULT_EXTERNAL_RAW_TEST_FILE}.", args.verbose)

if __name__ == '__main__':
    main()