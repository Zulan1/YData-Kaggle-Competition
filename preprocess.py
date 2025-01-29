import os
import pandas as pd
import constants as cons
import os

from sklearn.model_selection import train_test_split

from app.helper_functions import split_dataset_Xy, combine_Xy, save_data_for_training, log

from app.helper_functions import align_columns, clean_data, split_dataset_Xy, combine_Xy, save_data_for_training, log, encode_data
from app.argparser import get_preprocessing_args

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
    
    #2. Clean the data
    df = clean_data(df, columns_to_drop=cons.COLUMNS_TO_DROP, datetime_column=cons.DATETIME_COLUMN)
      
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