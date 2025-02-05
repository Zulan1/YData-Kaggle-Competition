import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from app.argparser import get_preprocessing_args
from app.feature_engineering import (
    add_first_session_feature, add_product_viewed_before,
    add_session_within_last_hour, flag_high_volume_users
)
from app.helper_functions import (
    extract_time_features, get_imputer, get_ohe, log, one_hot_encode,
    process_datetime, save_data_for_test, save_imputer_to_file,
    save_ohe_to_file, transform_categorical_columns, save_data_for_holdout,
    save_data_for_training, save_data_for_validation
)
import constants as cons

def split_by_user(df, split_ratios=(0.2, 0.2, 0.6)):
    """Split DataFrame into train/val/test while keeping user behavior balanced."""
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")
    
    # Get user stats
    user_stats = df.groupby('user_id').agg({
        'session_id': 'nunique',  # Number of unique sessions per user
        'is_click': 'sum'  # Number of clicks per user
    }).reset_index()
    
    # Add a behavior category column
    user_stats['click_distribution'] = user_stats.apply(
        lambda x: f"{x['session_id']}_sessions_{x['is_click']}_clicks", axis=1
    )
    
    # Stratify by click distribution
    behavior_groups = user_stats.groupby('click_distribution')['user_id'].apply(list).to_dict()
    
    first_users, second_users, third_users = [], [], []
    
    # Split each behavior group according to ratios
    for behavior, users in behavior_groups.items():
        users = np.array(users)
        np.random.seed(cons.RANDOM_STATE)
        np.random.shuffle(users)
        
        n_users = len(users)
        n_first = int(round(split_ratios[0] * n_users))
        n_second = int(round(split_ratios[1] * n_users))
        
        if n_first + n_second > n_users:
            n_second = n_users - n_first
            
        first_users.extend(users[:n_first])
        second_users.extend(users[n_first:n_first + n_second])
        third_users.extend(users[n_first + n_second:])
    
    # Create the splits
    first_split = df[df['user_id'].isin(first_users)]
    second_split = df[df['user_id'].isin(second_users)]
    third_split = df[df['user_id'].isin(third_users)]

    
    return first_split, second_split, third_split

def add_features(df):
    """Add all feature columns to the dataframe."""
    df = extract_time_features(df)
    df = add_product_viewed_before(df)
    df = add_session_within_last_hour(df)
    df = add_first_session_feature(df)
    return df

def preprocess_towards_training(df):
    """Preprocess training data and return processed df and fitted OHE."""
    # Process datetime and add features
    df = process_datetime(df)
    df = add_features(df)
    
    # Remove unnecessary columns
    df = df.drop(columns=cons.INDEX_COLUMNS + [cons.DATETIME_COLUMN])
    
    # One-hot encode
    df, ohe = one_hot_encode(df)
    df = df.drop(columns=cons.COLUMNS_TO_DROP)
    
    return df, ohe

def remove_outliers(df):
    """Remove outliers from the dataframe by dropping users with high session counts."""
    # Count sessions per user
    session_counts = df['user_id'].value_counts()
    
    # Find cutoff point at 99th percentile
    cutoff = np.percentile(session_counts, 95)
    
    # Get users below the 99th percentile
    valid_users = session_counts[session_counts <= cutoff].index
    
    # Filter dataframe to only include valid users
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    return df_filtered

def preprocess_towards_evaluation(df, ohe, imputer):
    """Preprocess validation/test data using fitted transformers."""
    # Process datetime and add features
    df = process_datetime(df)
    df = add_features(df)
    
    # Remove unnecessary columns
    df = df.drop(columns=cons.INDEX_COLUMNS + [cons.DATETIME_COLUMN])
    
    # Apply transformations
    df[cons.COLUMNS_TO_IMPUTE] = imputer.transform(df[cons.COLUMNS_TO_IMPUTE])
    df = transform_categorical_columns(df, ohe)
    df = df.drop(columns=cons.COLUMNS_TO_DROP)
    
    return df

def clean_data(df):
    """Remove rows with NA values and duplicates."""
    cols_to_check = [col for col in df.columns if col not in cons.COLUMNS_TO_DROP]
    df = df.dropna(subset=cols_to_check)
    df = df.drop_duplicates()
    return df

def main():
    args = get_preprocessing_args()
    output_path = os.path.join(args.output_path, f"preprocess_{args.run_id}")
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    log(f"Processing file: {args.input_path}", args.verbose)
    df = pd.read_csv(args.input_path)
    if df.empty:
        raise ValueError("The input file is empty.")

    if not args.test:
        # Training mode
        df = clean_data(df)
        df_train, df_val, df_holdout = split_by_user(df, cons.TRAIN_TEST_VAL_SPLIT)

        # Print stats if verbose
        if args.verbose:
            total = len(df_train) + len(df_val) + len(df_holdout)
            print("\nSplit sizes:")
            print(f"Train:      {len(df_train) / total:.4f}")
            print(f"Validation: {len(df_val) / total:.4f}")
            print(f"Holdout:    {len(df_holdout) / total:.4f}")
            
            print("\nClick rates:")
            print(f"Train:      {df_train['is_click'].mean():.4f}")
            print(f"Validation: {df_val['is_click'].mean():.4f}")
            print(f"Holdout:    {df_holdout['is_click'].mean():.4f}")

        # Fit and save imputer
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(df_train[cons.COLUMNS_TO_IMPUTE])
        save_imputer_to_file(imputer, os.path.join(output_path, cons.DEFAULT_IMPUTER_FILE), args.verbose)

        # Process and save training data
        df_train, ohe = preprocess_towards_training(df_train)
        save_ohe_to_file(ohe, os.path.join(output_path, cons.DEFAULT_OHE_FILE), args.verbose)
        save_data_for_training(df_train, output_path)

        # Process and save validation data
        df_val = preprocess_towards_evaluation(df_val, ohe, imputer)
        save_data_for_validation(df_val, output_path)

        # Save holdout data
        save_data_for_holdout(df_holdout, output_path)

        if args.verbose:
            print(f"Saved preprocessed data to {output_path}")
    
    else:
        # Test mode
        ohe = get_ohe(output_path)
        imputer = get_imputer(output_path)
        df = preprocess_towards_evaluation(df, ohe, imputer)
        save_data_for_test(df, output_path)
        log(f"Test set saved to {output_path}.", args.verbose)

if __name__ == '__main__':
    main()