import os
from typing import Tuple

import numpy as np
import pandas as pd
import pickle
from app.argparser import get_preprocessing_args
from app.helper_functions import (
    log, save_data_for_test,
    save_data_for_training, save_data_for_validation, get_data
)
import constants as cons
from splitting import split_by_user
from transformer import DataTransformer

def preprocess_towards_training(df):
    """Preprocess training data and retun the fitted DataTransformer."""
    transformer = DataTransformer()
    transformer.fit(df)
    df = transformer.transform(df)
    df = df.drop(columns=cons.INDEX_COLUMNS + [cons.DATETIME_COLUMN])
    return df, transformer

def preprocess_towards_evaluation(df, transformer):
    """Preprocess validation/test data using fitted transformers."""
    df = transformer.transform(df)
    df = df.drop(columns=cons.INDEX_COLUMNS + [cons.DATETIME_COLUMN])
    return df

def clean_data(df):
    """Remove rows with NA values and duplicates."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def save_transformer_to_file(transformer, file_path, verbose):
    """Save the transformer to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(transformer, f)
    if verbose:
        print(f"Transformer saved to {file_path}")

def get_transformer(transformer_path: str):
    """Load trained transformer from pickle file."""
    with open(transformer_path, 'rb') as f:
        return pickle.load(f)


def main():
    args = get_preprocessing_args()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    df = get_data(args.input_path, args.verbose)

    if args.mode == 'train':
        df = df.drop(columns=cons.COLUMNS_TO_DROP)
        df = clean_data(df)
        df_train, df_val, df_test = split_by_user(df)
        df_train, transformer = preprocess_towards_training(df_train)
        df_val = preprocess_towards_evaluation(df_val, transformer)
        df_test = preprocess_towards_evaluation(df_test, transformer)
        transformer_path = os.path.join(output_path, cons.DEFAULT_TRANSFORMER_FILE)
        save_transformer_to_file(transformer, transformer_path, args.verbose)
        save_data_for_training(df_train, output_path)
        save_data_for_validation(df_val, output_path)
        save_data_for_test(df_test, output_path)

        if args.verbose:
            print(f"Saved preprocessed data to {output_path}")
    
    elif args.mode == 'test':
        transformer = get_transformer(args.transformer_path)
        df = preprocess_towards_evaluation(df, transformer)
        save_data_for_test(df, output_path)
        log(f"Test set saved to {output_path}.", args.verbose)

    else:
        raise ValueError("Invalid mode.")

if __name__ == '__main__':
    main()