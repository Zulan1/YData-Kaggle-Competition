import os
import pandas as pd
import constants as cons
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle
from typing import Tuple

from app.helper_functions import (
    log, process_datetime, extract_time_features,
    one_hot_encode, get_ohe, get_imputer,
    save_imputer_to_file, save_ohe_to_file, transform_categorical_columns, save_data_for_test
)
from app.argparser import get_preprocessing_args
from app.feature_engineering import add_product_viewed_before, add_session_within_last_hour, add_first_session_feature


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
        rng = np.random.default_rng(cons.RANDOM_STATE)
        rng.shuffle(user_ids)  # randomize ordering to avoid ordering biases
        
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

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_time_features(df)
    df = add_product_viewed_before(df)
    df = add_session_within_last_hour(df)
    df = add_first_session_feature(df)
    return df

def preprocess_towards_training(df):
    """Preprocess training data."""
    df = process_datetime(df)
    df = add_features(df)
    # Drop index columns and datetime before one-hot encoding
    df = df.drop(columns=cons.INDEX_COLUMNS)
    df = df.drop(columns=[cons.DATETIME_COLUMN])
    # Now perform one-hot encoding on the cleaned dataframe
    df, ohe = one_hot_encode(df)
    df = df.drop(columns=cons.COLUMNS_TO_DROP)
    return df, ohe

def preprocess_towards_evaluation(df, ohe, imputer):
    """Preprocess data for validation or test sets using pre-fitted transformers.
    
    This function applies the same preprocessing steps used in training to new data,
    using the one-hot encoder (ohe) and imputer that were fit on the training data.
    It is used for both validation data during training and for test data during inference.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess
        ohe (OneHotEncoder): Fitted one-hot encoder from training
        imputer (SimpleImputer): Fitted imputer from training
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for model evaluation
    """
    df = process_datetime(df)
    df = add_features(df)
    df = df.drop(columns=cons.INDEX_COLUMNS)
    df = df.drop(columns=[cons.DATETIME_COLUMN])
    imputed_cols = imputer.transform(df[cons.COLUMNS_TO_IMPUTE])
    df[cons.COLUMNS_TO_IMPUTE] = imputed_cols
    df = transform_categorical_columns(df, ohe)
    df = df.drop(columns=cons.COLUMNS_TO_DROP)
    return df

def clean_data(df):
    df = df.dropna(subset=[col for col in df.columns if col not in cons.COLUMNS_TO_DROP])
    df = df.drop_duplicates()
    return df

def save_data_for_holdout(df, output_path):
    holdout_features = df.drop(columns=cons.TARGET_COLUMN)
    holdout_features.to_csv(os.path.join(output_path, cons.DEFAULT_HOLDOUT_FEATURES_FILE), index=False)
    holdout_labels = df[cons.TARGET_COLUMN]
    holdout_labels.to_csv(os.path.join(output_path, cons.DEFAULT_HOLDOUT_LABELS_FILE), index=False)
    return

def save_data_for_training(df, output_path):
    df.to_csv(os.path.join(output_path, cons.DEFAULT_TRAIN_SET_FILE), index=False)
    return

def save_data_for_validation(df, output_path):
    df.to_csv(os.path.join(output_path, cons.DEFAULT_VAL_SET_FILE), index=False)
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

        df_val = preprocess_towards_evaluation(df_val, ohe, imputer)

        save_data_for_training(df_train, output_path)
        save_data_for_validation(df_val, output_path)

        
        if args.verbose:
            print(f"Saved preprocessed data to {output_path}")

    if test_mode:
        ohe = get_ohe(output_path)
        imputer = get_imputer(output_path)
        df = preprocess_towards_evaluation(df, ohe, imputer)
        save_data_for_test(df, output_path)

        log(f"Test set saved to {output_path}.", args.verbose)

if __name__ == '__main__':
    main()