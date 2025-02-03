import os
import pandas as pd
import constants as cons
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle

from app.helper_functions import log, process_datetime, extract_time_features
from app.argparser import get_preprocessing_args

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


def split_by_user(df, split_ratios=(0.2, 0.2, 0.6)):
    """
    Split DataFrame into three parts while preserving user behavior distributions across splits.
    
    For each user, this function counts the number of sessions (by counting rows) and computes the 
    total number of clicks (by summing the 'n_clicks' column), then groups users according to these
    aggregated metrics. The users in each stratification group are shuffled and split according
    to the provided ratios, ensuring that each split will have the same distribution of users with
    k sessions and j clicks while also guaranteeing that each user appears in exactly one split.
    
    Args:
        df: DataFrame containing at least 'user_id' and 'n_clicks' columns. The number of sessions
            per user is computed by counting rows per user.
        split_ratios: Tuple of 3 floats that sum to 1, representing the proportions of the three splits.
    
    Returns:
        Three DataFrames corresponding to the splits.
    """
    # Allow for floating-point imprecision when verifying that ratios sum to 1
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")
    
    # Aggregate per-user statistics:
    # - Count the number of sessions (number of rows per user)
    # - Compute the total number of clicks over all sessions for that user.
    user_stats = df.groupby('user_id', as_index=False).agg(
        n_sessions=('user_id', 'size'),
        n_clicks=('is_click', 'sum')
    )
    
    # Create stratification groups based on the aggregated values.
    # For example, a user with 3 sessions and 10 clicks will belong to the group "s3_c10".
    user_stats['strat_group'] = user_stats.apply(
        lambda x: f"s{x['n_sessions']}_c{x['n_clicks']}", axis=1
    )
    
    first_user_ids = []
    second_user_ids = []
    third_user_ids = []
    
    # For every unique stratification group, shuffle and split user_ids based on ratios.
    for _, group in user_stats.groupby('strat_group'):
        user_ids = group['user_id'].tolist()
        np.random.shuffle(user_ids)  # randomize ordering to avoid ordering biases
        
        total = len(user_ids)
        count_first = int(round(split_ratios[0] * total))
        count_second = int(round(split_ratios[1] * total))
        
        # Adjust for potential rounding issues
        if count_first + count_second > total:
            count_second = total - count_first
        
        first_user_ids.extend(user_ids[:count_first])
        second_user_ids.extend(user_ids[count_first:count_first + count_second])
        third_user_ids.extend(user_ids[count_first + count_second:])
    
    # Create masks for the original DataFrame to ensure each user appears only in one split.
    mask_first = df['user_id'].isin(first_user_ids)
    mask_second = df['user_id'].isin(second_user_ids)
    mask_third = df['user_id'].isin(third_user_ids)
    
    return df[mask_first], df[mask_second], df[mask_third]

def one_hot_encode(df):
    """Encode categorical columns using OneHotEncoder.
    Returns the encoded DataFrame and the encoder."""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = ohe.fit_transform(df[cons.CATEGORICAL])
    feature_names = ohe.get_feature_names_out(cons.CATEGORICAL)
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    df = df.drop(columns=cons.CATEGORICAL)  # Only drops categorical columns
    df = pd.concat([df, encoded_df], axis=1)
    return df, ohe

def get_ohe(input_path) -> OneHotEncoder:
    ohe_path = os.path.join(input_path, cons.DEFAULT_OHE_FILE)
    return pickle.load(open(ohe_path, 'rb'))

def get_imputer(input_path) -> SimpleImputer:
    imputer_path = os.path.join(input_path, cons.DEFAULT_IMPUTER_FILE)
    return pickle.load(open(imputer_path, 'rb'))

def save_imputer_to_file(imputer, path, verbose):
    """Save the Imputer to a file."""
    with open(path, 'wb') as f:
        pickle.dump(imputer, f)
    if verbose:
        print(f"Imputer saved to {path}")

def save_ohe_to_file(ohe, path, verbose):
    """Save the OneHotEncoder to a file."""
    with open(path, 'wb') as f:
        pickle.dump(ohe, f)
    if verbose:
        print(f"OneHotEncoder saved to {path}")

def save_data_for_prediction(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_time_features(df)
    return df

def preprocess_towards_training(df):
    """Preprocess training data."""
    df = process_datetime(df)
    df = feature_engineering(df)
    # Drop index columns and datetime before one-hot encoding
    df = df.drop(columns=cons.INDEX_COLUMNS)
    df = df.drop(columns=[cons.DATETIME_COLUMN])
    # Now perform one-hot encoding on the cleaned dataframe
    df, ohe = one_hot_encode(df)
    return df, ohe

def preprocess_towards_validation(df, ohe):
    """Preprocess validation data."""
    df = feature_engineering(df)
    df = transform_categorical_columns(df, ohe)
    df = df.drop(columns=cons.INDEX_COLUMNS)
    df = df.drop(columns=[cons.DATETIME_COLUMN])
    return df

def preprocess_towards_test(df, ohe, imputer):
    """Preprocess test data using fitted transformers."""
    # Process datetime if column exists
    if cons.DATETIME_COLUMN in df.columns:
        df = process_datetime(df)
        df = extract_time_features(df)

    imputed_cols = imputer.transform(df[cons.COLUMNS_TO_IMPUTE])
    df[cons.COLUMNS_TO_IMPUTE] = imputed_cols
    # Transform categorical columns after imputation
    df = transform_categorical_columns(df, ohe)
    
    # Convert to numpy array for imputation, then back to DataFrame
    
    # Drop unnecessary columns if they exist
    for col in cons.INDEX_COLUMNS + [cons.DATETIME_COLUMN]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def save_data_for_holdout(df, output_path):
    holdout_features = df.drop(columns=cons.TARGET_COLUMN)
    holdout_features.to_csv(os.path.join(output_path, cons.DEFAULT_HOLDOUT_FEATURES_FILE), index=False)
    holdout_labels = df[cons.TARGET_COLUMN]
    holdout_labels.to_csv(os.path.join(output_path, cons.DEFAULT_HOLDOUT_LABELS_FILE), index=False)
    return

def save_features_list_to_file(features_list, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(features_list, f)
    return

def save_data_for_training(df, output_path):
    df.to_csv(os.path.join(output_path, cons.DEFAULT_TRAIN_SET_FILE), index=False)
    return

def save_data_for_validation(df, output_path):
    df.to_csv(os.path.join(output_path, cons.DEFAULT_VAL_SET_FILE), index=False)
    return

def save_data_for_test(df, output_path):
    df.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_SET_FILE), index=False)
    return

def main():
    args = get_preprocessing_args()
    run_id = args.run_id
    train_mode = not args.test
    test_mode = args.test
    full_fn = args.input_path
    
    output_path = os.path.join(args.output_path, f"preprocess_{run_id}")
    imputer_path = os.path.join(output_path, cons.DEFAULT_IMPUTER_FILE)
    ohe_path = os.path.join(output_path, cons.DEFAULT_OHE_FILE)

    os.makedirs(output_path, exist_ok=True)
    
    log(f"Processing file: {full_fn}", args.verbose)
    df = pd.read_csv(full_fn)
    if df.empty:
        raise ValueError("The input file is empty.")
    

    if train_mode:
        df = df.drop(columns=cons.COLUMNS_TO_DROP)
        df = clean_data(df)

        df_train, df_val, df_holdout = split_by_user(df, cons.TRAIN_TEST_VAL_SPLIT)

        if args.verbose:

            print("\nSplit proportions (should be approximately 0.6, 0.2, 0.2):")
            print(f"Train set:      {len(df_train) / (len(df_train) + len(df_val) + len(df_holdout)):.4f}")
            print(f"Validation set: {len(df_val) / (len(df_train) + len(df_val) + len(df_holdout)):.4f}")
            print(f"Holdout set:    {len(df_holdout) / (len(df_train) + len(df_val) + len(df_holdout)):.4f}")
            print("\nClick rates (should be similar across splits):")
            print(f"Train set:      {df_train[cons.TARGET_COLUMN].mean():.4f}")
            print(f"Validation set: {df_val[cons.TARGET_COLUMN].mean():.4f}")
            print(f"Holdout set:    {df_holdout[cons.TARGET_COLUMN].mean():.4f}")

        save_data_for_holdout(df_holdout, output_path)

        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(df_train[cons.COLUMNS_TO_IMPUTE])
        save_imputer_to_file(imputer, imputer_path, args.verbose)

        df_train, ohe = preprocess_towards_training(df_train)
        save_ohe_to_file(ohe, ohe_path, args.verbose)

        df_val = preprocess_towards_validation(df_val, ohe)

        save_data_for_training(df_train, output_path)
        save_data_for_validation(df_val, output_path)

        
        if args.verbose:
            print(f"Saved preprocessed data to {output_path}")

    if test_mode:
        ohe = get_ohe(output_path)
        imputer = get_imputer(output_path)
        df = preprocess_towards_test(df, ohe, imputer)
        save_data_for_test(df, output_path)

        log(f"Test set saved to {output_path}.", args.verbose)

if __name__ == '__main__':
    main()