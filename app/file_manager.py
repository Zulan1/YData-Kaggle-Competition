import os
import pandas as pd
import constants as cons
import pickle
from transformer import DataTransformer
from typing import Any
from app.helper_functions import log

def save_data_for_test(df: pd.DataFrame, output_path: str) -> None:
    """Save holdout features and labels to separate CSV files."""
    features = df.drop(columns=cons.TARGET_COLUMN)
    labels = df[cons.TARGET_COLUMN]
    features.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_FEATURES_FILE), index=False)
    labels.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_LABELS_FILE), index=False)
    return

def save_data_for_validation(df: pd.DataFrame, output_path: str) -> None:
    """Save validation features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_VAL_SET_FILE), index=False)
    return

def save_data_for_training(df: pd.DataFrame, output_path: str) -> None:
    """Save training features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_TRAIN_SET_FILE), index=False)
    return

def save_predictions(df, output_path, verbose):
    predictions_path = os.path.join(output_path, cons.DEFAULT_PREDICTIONS_FILE)
    """Save predictions to a CSV file."""
    df.to_csv(predictions_path, index=False)
    return

def get_model(model_path: str, verbose: bool) -> Any:
    """Get a model from a pickle file."""
    log("Loading model", verbose)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    log(f"Model type: {model.__class__.__name__}", verbose)
    return model

def save_model(model: Any, output_path: str) -> None:
    """Save a model to a pickle file."""
    model_path = os.path.join(output_path, cons.DEFAULT_MODEL_FILE)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return

def get_transformer(transformer_path: str) -> DataTransformer:
    """Get a transformer from a pickle file."""
    with open(transformer_path, 'rb') as f:
        return pickle.load(f)

def save_transformer(transformer: DataTransformer, output_path: str, verbose: bool) -> None:
    transformer_path = os.path.join(output_path, cons.DEFAULT_TRANSFORMER_FILE)
    """Save a transformer to a pickle file."""
    with open(transformer_path, 'wb') as f:
        pickle.dump(transformer, f)
    log(f"Transformer saved to {transformer_path}", verbose)
    return

def get_data(input_path: str, verbose: bool) -> pd.DataFrame:
    """Get data from a CSV file."""
    log(f"Loading data from {input_path}", verbose)
    return pd.read_csv(input_path)