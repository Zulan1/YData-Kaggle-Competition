import os
import pandas as pd
import constants as cons
import pickle
from transformer import DataTransformer
from typing import Any
from app.helper_functions import log

def save_data_for_test(df: pd.DataFrame, output_path: str, verbose: bool) -> None:
    """Save holdout features and labels to separate CSV files."""
    features = df.drop(columns=cons.TARGET_COLUMN)
    labels = df[cons.TARGET_COLUMN]
    features.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_FEATURES_FILE), index=False)
    dtypes_dict = features.dtypes.astype(str).to_dict()
    with open(os.path.join(output_path, cons.DEFAULT_TEST_DTYPES_FILE), 'wb') as f:
        pickle.dump(dtypes_dict, f)
    labels.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_LABELS_FILE), index=False)
    return

def save_data_for_external_test(df: pd.DataFrame, output_path: str, verbose: bool) -> None:
    """Save holdout features and labels to separate CSV files."""
    features = df
    features.to_csv(os.path.join(output_path, cons.DEFAULT_EXTERNAL_TEST_FEATURES_FILE), index=False)
    dtypes_dict = features.dtypes.astype(str).to_dict()
    with open(os.path.join(output_path, cons.DEFAULT_EXTERNAL_TEST_DTYPES_FILE), 'wb') as f:
        pickle.dump(dtypes_dict, f)

def save_data_for_validation(df: pd.DataFrame, output_path: str) -> None:
    """Save validation features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_VAL_SET_FILE), index=False)
    dtypes_dict = df.dtypes.astype(str).to_dict()
    with open(os.path.join(output_path, cons.DEFAULT_VAL_DTYPES_FILE), 'wb') as f:
        pickle.dump(dtypes_dict, f)
    return

def save_data_for_training(df: pd.DataFrame, output_path: str) -> None:
    """Save training features and labels to separate CSV files."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_TRAIN_SET_FILE), index=False)
    dtypes_dict = df.dtypes.astype(str).to_dict()
    with open(os.path.join(output_path, cons.DEFAULT_TRAIN_DTYPES_FILE), 'wb') as f:
        pickle.dump(dtypes_dict, f)
    return

def get_val_set(input_path: str) -> pd.DataFrame:
    """Get validation set from a CSV file."""
    df_val = pd.read_csv(os.path.join(input_path, cons.DEFAULT_VAL_SET_FILE))
    df_val = df_val.copy()
    with open(os.path.join(input_path, cons.DEFAULT_VAL_DTYPES_FILE), 'rb') as f:
        dtypes_dict = pickle.load(f)
    for col, dtype in dtypes_dict.items():
        df_val[col] = df_val[col].astype(dtype)
    return df_val

def get_train_set(input_path: str) -> pd.DataFrame:
    """Get training set from a CSV file."""
    df_train = pd.read_csv(os.path.join(input_path, cons.DEFAULT_TRAIN_SET_FILE))
    df_train = df_train.copy()
    with open(os.path.join(input_path, cons.DEFAULT_TRAIN_DTYPES_FILE), 'rb') as f:
        dtypes_dict = pickle.load(f)
    for col, dtype in dtypes_dict.items():
        df_train[col] = df_train[col].astype(dtype)
    return df_train

def save_full_processed_training(df: pd.DataFrame, output_path: str) -> None:
    """Save full processed training data to a CSV file."""
    df.to_csv(os.path.join(output_path, cons.DEFAULT_FULL_PROCESSED_TRAINING_FILE), index=False)
    dtypes_dict = df.dtypes.astype(str).to_dict()
    with open(os.path.join(output_path, cons.DEFAULT_FULL_PROCESSED_TRAINING_DTYPES_FILE), 'wb') as f:
        pickle.dump(dtypes_dict, f)
    return

def load_full_processed_training(input_path: str) -> pd.DataFrame:
    """Load full processed training data from a CSV file."""
    df_full = pd.read_csv(os.path.join(input_path, cons.DEFAULT_FULL_PROCESSED_TRAINING_FILE))
    df_full = df_full.copy()
    with open(os.path.join(input_path, cons.DEFAULT_FULL_PROCESSED_TRAINING_DTYPES_FILE), 'rb') as f:
        dtypes_dict = pickle.load(f)
    for col, dtype in dtypes_dict.items():
        df_full[col] = df_full[col].astype(dtype)
    return df_full

def get_test_features(test_features_path: str, test_dtypes_path: str) -> pd.DataFrame:
    """Get test features from a CSV file."""
    
    df_test = pd.read_csv(test_features_path)
    df_test = df_test.copy()

    with open(test_dtypes_path, 'rb') as f:
        dtypes_dict = pickle.load(f)

    for col, dtype in dtypes_dict.items():
        df_test[col] = df_test[col].astype(dtype)

    return df_test

def save_full_model(model, output_path):
    model_path = os.path.join(output_path, cons.DEFAULT_FULL_MODEL_FILE)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return

def save_predictions(df, output_path, verbose):
    predictions_path = os.path.join(output_path, cons.DEFAULT_PREDICTIONS_FILE)
    """Save predictions to a CSV file without the 'is_click' header."""
    df.to_csv(predictions_path, index=False, header=False)
    if verbose:
        log(f"[file_manager.py] Predictions saved to {predictions_path} (header excluded)", verbose)
    return

def save_predicted_probabilities(df, output_path, verbose):
    probabilities_path = os.path.join(output_path, cons.DEFAULT_PREDICTED_PROBABILITIES_FILE)
    """Save predicted probabilities to a CSV file."""
    df.to_csv(probabilities_path, index=False)
    if verbose:
        log(f"[file_manager.py] Predicted probabilities saved to {probabilities_path}", verbose)
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
    log(f"[file_manager.py] Transformer saved to {transformer_path}", verbose)
    return

def get_data(input_path: str, verbose: bool) -> pd.DataFrame:
    """Get data from a CSV file."""
    log(f"[file_manager.py] Loading data from {input_path}", verbose)
    return pd.read_csv(input_path)
