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
    dtypes_dict = features.dtypes.astype(str).to_dict()
    with open(os.path.join(output_path, cons.DEFAULT_TEST_DTYPES_FILE), 'wb') as f:
        pickle.dump(dtypes_dict, f)
    labels.to_csv(os.path.join(output_path, cons.DEFAULT_TEST_LABELS_FILE), index=False)
    print("dtypes_dict for save", dtypes_dict)
    return

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
    print("dtypes_dict for save", dtypes_dict)
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

def get_test_features(input_path: str) -> pd.DataFrame:
    """Get test features from a CSV file."""
    df_test = pd.read_csv(os.path.join(input_path, cons.DEFAULT_TEST_FEATURES_FILE))
    df_test = df_test.copy()
    with open(os.path.join(input_path, cons.DEFAULT_TEST_DTYPES_FILE), 'rb') as f:
        dtypes_dict = pickle.load(f)
    for col, dtype in dtypes_dict.items():
        df_test[col] = df_test[col].astype(dtype)
    print("df_test dtypes loaded", df_test.dtypes)
    return df_test

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