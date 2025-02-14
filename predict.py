"""
Predict module for generating predictions.

This module loads a trained model, reads test features, generates predictions and predicted probabilities,
and then saves these results to disk.
"""

from app.argparser import get_predict_args
import pandas as pd
import constants as cons
from app.file_manager import get_model, get_test_features, save_predictions, save_predicted_probabilities

def get_predicted_probabilities(model, df):
    """
    Generate predicted probabilities for the target column.
    
    Args:
        model: Trained model with a predict_proba method.
        df (pd.DataFrame): DataFrame with test features.

    Returns:
        pd.DataFrame: DataFrame of predicted probabilities.
    """
    return pd.DataFrame(
        model.predict_proba(df),
        index=df.index,
        columns=[cons.TARGET_COLUMN]
    )

def get_predictions(model, df):
    """
    Generate predictions for the target column.
    
    Args:
        model: Trained model with a predict method.
        df (pd.DataFrame): DataFrame with test features.

    Returns:
        pd.DataFrame: DataFrame of predictions.
    """

    return pd.DataFrame(
        model.predict(df),
        index=df.index,
        columns=[cons.TARGET_COLUMN]
    )

def main():
    args = get_predict_args()
    model = get_model(args.model_path, args.verbose)
    df = get_test_features(args.test_features_path, args.test_dtypes_path)

    if args.verbose:
        print(f"\n[predict.py] Loading model from {args.model_path}...")
        print(f"[predict.py] Loading test features from {args.test_features_path}...")
        print(f"[predict.py] Loading test dtypes from {args.test_dtypes_path}...")
        print(f"[predict.py] Features shape: {df.shape}")
        print(f"\n[predict.py] Predicting {cons.TARGET_COLUMN}...")

    predictions = get_predictions(model, df)
    save_predictions(predictions, args.output_path, args.verbose)
    
    probabilities = get_predicted_probabilities(model, df)
    save_predicted_probabilities(probabilities, args.output_path, args.verbose)

if __name__ == '__main__':
    main()
