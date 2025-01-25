import os
import pandas as pd
import constants as cons

from sklearn.model_selection import train_test_split

from app.helper_functions import split_dataset_Xy, combine_Xy, save_data_for_training, log
from app.argparser import get_preprocessing_args


def clean_data(df):
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
    df = df.drop_duplicates()

    #2. drop columns with high percentage of missing values
    df = df.drop(columns=cons.COLUMNS_TO_DROP)

    #3. Drop missing values
    df = df.dropna()

    #4. Encode categorical values using one-hot encoding (dummies)
    # Create dummies for these columns
    df = pd.get_dummies(df, columns=cons.CATEGORIAL, drop_first=True)

    #5. Convert DateTime into DateTime object and sort by DateTime so data is chronological:
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    if df['DateTime'].isna().sum() > 0:
        raise ValueError("Invalid DateTime entries found during preprocessing.")
    df = df.sort_values('DateTime')

    #6. Extract hour feature and weekday features from DateTime column and drop the original DateTime (save it for later)
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek # Monday=0, Tuesday=1, Wednesday=3, Thirsday=4, Friday=5, Saturday = 6
    df['DateTime'] = df['DateTime'].dt.tz_localize('Europe/London').dt.tz_convert('UTC').astype('int64') // 1e9

    return df

def main():
    args = get_preprocessing_args()

    full_fn = args.input_path
    log(f"Processing file: {full_fn}", args.verbose)

    #1. load csv file
    df = pd.read_csv(full_fn)

    #2. Clean the data
    df = clean_data(df)
      
    if not args.test: 
        # Split the data into features (X) and target (y)
        X, y = split_dataset_Xy(df)

        # First split: Train (60%) and Temp (40% for validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=42)

        # Second split: Temp -> Validation (20%) and Test (20%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        # Combine X and y back into DataFrames for train, validation, and test
        train = combine_Xy(X_train, y_train)
        val = combine_Xy(X_val, y_val)
        test = combine_Xy(X_test, y_test)

        log(f"Training set created with {len(train)} samples.", args.verbose)
        log(f"Validation set created with {len(val)} samples.", args.verbose)
        log(f"Test set created with {len(test)} samples.", args.verbose)

        # Save the datasets using the helper function
        save_data_for_training(train, val, test, path=args.out_path)

    else:
        # Process the test set (for prediction)
        df.to_csv(os.path.join(args.csv_path, cons.DEFAULT_TEST_SET_FILE), index=False)
        log(f"Test set saved to {cons.DEFAULT_TEST_SET_FILE}.", args.verbose)

if __name__ == '__main__':
    main()
