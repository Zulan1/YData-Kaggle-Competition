import argparse
import pandas as pd
import cons
from sklearn.model_selection import train_test_split

import sys
import os

# Add the 'app/' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), './app/'))

import utilities as utils

def main():
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

    #2. Remove duplicates
    df = df.copy().drop_duplicates()
    
    #3. drop columns with high percentage of missing values
    columns_to_drop = ['product_category_2', 'city_development_index']
    df = df.copy().drop(columns=columns_to_drop)

    #4. Drop missing values
    df = df.copy().dropna()

    #5. Create new features:
    sessions_per_user = df.groupby('user_id')['session_id'].nunique().reset_index()
    sessions_per_user.rename(columns={'session_id': 'sessions_per_user'}, inplace=True)
    df = df.copy().merge(sessions_per_user, on='user_id', how='left')

    # 6. Encode categorical values using one-hot encoding (dummies)
    # Create dummies for these columns
    df = pd.get_dummies(df.copy(), columns=cons.CATEGORIAL, drop_first=True)

    #7. Convert DateTime into DateTime object and sort by DateTime so data is chronological, set DateTime as index and fill missing values:
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.copy().sort_values('DateTime')
    df.set_index('DateTime', inplace=True)
    df['DateTime'] = df['DateTime'].ffill().bfill()

    #8. Extract hour feature from DateTime column and drop the original DateTime (save it for later)
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek # Monday=0, Tuesday=1, Wednesday=3, Thirsday=4, Friday=5, Saturday = 6
    df = df.copy().drop(columns=['DateTime'])

    if args.verbose:
        utils.display_unique(df)

    #9. Split the data into training and validation sets
    # Perform stratified split (80/20)
    # Define features and target column
    if not args.test:
        X = df.drop(columns=['is_click'])  # Features
        y = df['is_click']                # Target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Combine X and y back into DataFrames for train and test
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        # Verify the stratified split
        if args.verbose:
            print(f"Train size: {len(train)}")
            print(f"Test size: {len(test)}")
            print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
            print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

        # Save to CSV (optional)
        train.to_csv(args.csv_path + '/' + cons.DEFAULT_PROCESSED_TRAIN_FILE, index=False)
        test.to_csv(args.csv_path + '/' +cons.DEFUALT_PROCESSED_TEST_FILE, index=False)
    else:
        df.to_csv(args.csv_path + '/' +cons.DEFUALT_PROCESSED_TEST_FILE, index=False)



if __name__ == '__main__':
    main()
