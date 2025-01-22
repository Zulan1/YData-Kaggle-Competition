import argparse
import pandas as pd
import constants as cons
from sklearn.model_selection import train_test_split, StratifiedKFold

import sys
import os

# Add the 'app/' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), './app/'))

import utilities as utils
from helper_functions import split_dataset_Xy, combine_Xy, save_data_for_training


def clean_data (df):
    """Clean the data:
    1. Remove duplicates
    2. Drop columns with high percentage of missing values
    3. Drop missing values
    4. Encode categorical values using one-hot encoding (dummies)
    5. Convert DateTime into DateTime object and sort by DateTime so data is chronological
    6. Extract hour and weekday features from DateTime column and drop the original DateTime

    Args:
        df (pd.DataFrame): The DataFrame containing the fold data.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    #1. Remove duplicates
    df = df.copy().drop_duplicates()

    #2. drop columns with high percentage of missing values
    columns_to_drop = ['product_category_2', 'city_development_index']
    df = df.copy().drop(columns=columns_to_drop)

    #3. Drop missing values
    df = df.copy().dropna()

    #4. Encode categorical values using one-hot encoding (dummies)
    # Create dummies for these columns
    df = pd.get_dummies(df.copy(), columns=cons.CATEGORIAL, drop_first=True)

    #5. Convert DateTime into DateTime object and sort by DateTime so data is chronological:
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.copy().sort_values('DateTime')

    #6. Extract hour feature and weekday features from DateTime column and drop the original DateTime (save it for later)
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek # Monday=0, Tuesday=1, Wednesday=3, Thirsday=4, Friday=5, Saturday = 6
    df = df.copy().drop(columns=['DateTime'])

    return df


def add_cum_ctr (df):
    """Add cumulative CTR feature to the data:
    Args:
        df (pd.DataFrame): The DataFrame containing the fold data.
    Returns:
        pd.DataFrame: The DataFrame with the added features.
    """
    df['cum_ctr'] = df.groupby('user_id')['is_click'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    )
    return df

def add_sessions_per_user (df):
    """Add sessions per user feature to the data:
    Args:
        df (pd.DataFrame): The DataFrame containing the fold data.
    Returns:
        pd.DataFrame: The DataFrame with the added features.
    """
    sessions_per_user = df.groupby('user_id')['session_id'].nunique().reset_index()
    sessions_per_user.rename(columns={'session_id': 'sessions_per_user'}, inplace=True)
    df = df.copy().merge(sessions_per_user, on='user_id', how='left')
    df = df.copy().drop(columns=['session_id'])
    return df


def add_engineered_features (df):
    """Add engineered features to the data by combining multiple feature engineering steps.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to engineer features for.
        
    Returns:
        pd.DataFrame: The DataFrame with all engineered features added, including:
            - Cumulative click-through rate per user
            - Number of sessions per user
    """
    df = add_cum_ctr(df)
    df = add_sessions_per_user(df)
    return df



def main():
    #Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', default=cons.DATA_PATH, type=str, help='Path to input CSV file')
    parser.add_argument('--verbose', default=True, type=bool, help='Print additional information')
    parser.add_argument('--test', default=False, type=bool, help='Run on test rather on train dataset')
    parser.add_argument('--filename', default=cons.DEFAULT_RAW_TRAIN_FILE, type=str, help='CSV filename to load')

    args = parser.parse_args()

    full_fn = args.csv_path + '/' + args.filename
    if args.verbose:
        print(f"Processing file: {full_fn}")
    # Add preprocessing steps here

    #1. load csv file
    df = pd.read_csv(full_fn)

    #2. Clean the data
    df = clean_data(df)
      
    
    if not args.test: 
        #Split the data into training and validation sets, and preprocess each set
        # Perform stratified split (80/20)
        # Define features and target column
        X, y = split_dataset_Xy(df)

        
        # Split the data into training and test sets:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cons.DEFAULT_TEST_SIZE, stratify=y, random_state=42)

        # Perform stratified k-fold split on the training set
        skf = StratifiedKFold(n_splits=cons.DEFAULT_N_FOLDS, shuffle=True, random_state=42)
        fold_indices = list(skf.split(X_train, y_train))

        # Create DataFrames for each fold
        folds = []
        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            # Create DataFrame for the training and validation sets:
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Concatenate X and y fro train and validation each validation fold, and add engineered features
            # Feature engineering is done at this stage only, to avoid data leakage
            train_fold = add_engineered_features(combine_Xy(X_train_fold, y_train_fold))
            val_fold = add_engineered_features(combine_Xy(X_val_fold, y_val_fold))
            folds.append((train_fold, val_fold))

            if args.verbose:
                print(f"Fold {fold + 1} created with {len(train_fold)} training samples and {len(val_fold)} validation samples.")

        # Create DataFrame for the train and test sets, and add engineered features.
        # Here we use againg the train without folds, after optimizing the hyper-parameters we will use the whole set for training
        train_full = add_engineered_features(combine_Xy(X_train, y_train))
        test = add_engineered_features(combine_Xy(X_test, y_test))
        if args.verbose:
            print(f"Test set created with {len(test)} samples.")

        # Save the folds and test set to CSV files
        save_data_for_training(folds, train_full, test)

    else:
        # Process the test set (for prediction)
        # Add engineered features (we assume feature engineering does not require the target column)
        df = add_engineered_features(df)
        df.to_csv(args.csv_path + '/' +cons.DEFUALT_PROCESSED_TEST_FILE, index=False)



if __name__ == '__main__':
    main()
