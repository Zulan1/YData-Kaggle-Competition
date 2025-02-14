from app.argparser import get_preprocessing_args
from app.file_manager import (
    save_data_for_test,
    save_data_for_training, save_data_for_validation, get_data, save_transformer, get_transformer, save_data_for_external_test, save_full_processed_training
)
import pandas as pd
from transformer import DataTransformer
from splitting import DataSplitter
from data_cleaning import DataCleaner

def preprocess_towards_training(df, verbose=False):
    """Preprocess training data and return the fitted DataTransformer."""
    transformer = DataTransformer(verbose=verbose)
    transformer.fit(df)
    df = transformer.transform(df)
    return df, transformer

def preprocess_towards_evaluation(df, transformer):
    """Preprocess validation/test data using fitted transformers."""
    df = transformer.transform(df)
    return df

def create_splitter(verbose):
    return DataSplitter(
        verbose=verbose
    )

def main():
    args = get_preprocessing_args()

    if args.verbose:
        print(f"\n[preprocess.py] Preprocessing data from {args.csv_for_preprocessing} to {args.output_path}.")
        print(f"[preprocess.py] Running in mode: {args.mode}")

    output_path = args.output_path
    df = get_data(args.csv_for_preprocessing, args.verbose)
    splitter = create_splitter(args.verbose)

    if args.verbose:
        print(f"[preprocess.py] Number of rows before cleaning: {len(df)}")

    cleaner = DataCleaner(verbose=args.verbose, mode=args.mode)
    df = cleaner.clean_data(df)

    if args.verbose:
        print(f"[preprocess.py] Data cleaning completed. {len(df)} rows remaining.")

    if args.mode == 'train':

        if args.limit_data:
            print("[preprocess.py] Limiting data to 1500 rows for testing.")
            df = df[:1500]

        df_train, df_val, df_test = splitter.split_data(df)

        if args.verbose:
            print("\n[preprocess.py] Preprocessing of data started...")
        df_train, transformer = preprocess_towards_training(df_train, args.verbose)

        if args.verbose:
            print("[preprocess.py] Preprocessing of training data completed.")

        df_val = preprocess_towards_evaluation(df_val, transformer)
        if args.verbose:
            print("[preprocess.py] Preprocessing of validation data completed.")

        df_test = preprocess_towards_evaluation(df_test, transformer)
        if args.verbose:
            print("[preprocess.py] Preprocessing of test data completed.")
        
        df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)

        save_transformer(transformer, output_path, args.verbose)
        save_data_for_training(df_train, output_path)
        save_data_for_validation(df_val, output_path)
        save_data_for_test(df_test, output_path, args.verbose)
        save_full_processed_training(df_full, output_path)

        if args.verbose:
            print(f"\n[preprocess.py] Preprocessing of data completed. Data saved to {output_path}.")
    
    elif args.mode == 'inference':
        transformer = get_transformer(args.transformer_path)
        if args.limit_data:
            df = df[:1500]
        df = preprocess_towards_evaluation(df, transformer)
        save_data_for_external_test(df, output_path, args.verbose)
        if args.verbose:
            print(f"[preprocess.py] Test set saved to {output_path}.")

    else:
        raise ValueError("Invalid mode.")

if __name__ == '__main__':
    main()