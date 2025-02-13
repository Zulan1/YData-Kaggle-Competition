from app.argparser import get_preprocessing_args
from app.file_manager import (
    save_data_for_test,
    save_data_for_training, save_data_for_validation, get_data, save_transformer, get_transformer, save_data_for_external_test
)
import constants as cons
from splitting import split_data
from transformer import DataTransformer

def preprocess_towards_training(df, verbose=False):
    """Preprocess training data and retun the fitted DataTransformer."""
    transformer = DataTransformer(verbose=verbose)
    transformer.fit(df)
    df = transformer.transform(df)
    return df, transformer

def preprocess_towards_evaluation(df, transformer):
    """Preprocess validation/test data using fitted transformers."""
    df = transformer.transform(df)
    return df

def clean_data(df):
    """Remove rows with NA values and duplicates."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def main():
    args = get_preprocessing_args()

    if args.verbose:
        print(f"\nPreprocessing data from {args.csv_for_preprocessing} to {args.output_path}.")

    output_path = args.output_path
    df = get_data(args.csv_for_preprocessing, args.verbose)

    if args.mode == 'train':
        df = df.drop(columns=cons.COLUMNS_TO_DROP)
        df = clean_data(df)

        if args.limit_data:
            print("Limiting data to 1500 rows for testing.")
            df = df[:1500]

        df_train, df_val, df_test = split_data(df, verbose=args.verbose)

        if args.verbose:
            print("\nPreprocessing of data started...")
        df_train, transformer = preprocess_towards_training(df_train, args.verbose)

        if args.verbose:
            print("Preprocessing of training data completed.")

        df_val = preprocess_towards_evaluation(df_val, transformer)
        if args.verbose:
            print("Preprocessing of validation data completed.")

        df_test = preprocess_towards_evaluation(df_test, transformer)
        if args.verbose:
            print("Preprocessing of test data completed.")

        save_transformer(transformer, output_path, args.verbose)
        save_data_for_training(df_train, output_path)
        save_data_for_validation(df_val, output_path)
        save_data_for_test(df_test, output_path, args.verbose)

        if args.verbose:
            print(f"\nPreprocessing of data completed. Data saved to {output_path}.")
    
    elif args.mode == 'inference':
        transformer = get_transformer(args.transformer_path)
        df = df.drop(columns=cons.COLUMNS_TO_DROP)
        df = df.dropna()
        df['is_click'] = df['is_click'].astype(int)
        if args.limit_data:
            df = df[:5000]
        df = preprocess_towards_evaluation(df, transformer)
        save_data_for_external_test(df, output_path, args.verbose)
        if args.verbose:
            print(f"Test set saved to {output_path}.")

    else:
        raise ValueError("Invalid mode.")

if __name__ == '__main__':
    main()