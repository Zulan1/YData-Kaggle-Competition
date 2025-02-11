from app.argparser import get_predict_args
import pandas as pd
import pickle
import config as conf
import constants as cons
import os
from app.helper_functions import log
from app.file_manager import get_model, get_test_features, save_predictions, save_predicted_probabilities

def get_predicted_probabilities(model, df):
    return pd.DataFrame(
        model.predict_proba(df)[:, 1],
        index=df.index,
        columns=[cons.TARGET_COLUMN]
    )

def get_predictions(model, df):
    return pd.DataFrame(
        model.predict(df),
        index=df.index,
        columns=[cons.TARGET_COLUMN]
    )

def main():
    args = get_predict_args()
    model = get_model(args.model_path, args.verbose)
    df = get_test_features(args.input_path)
    os.makedirs(args.output_path, exist_ok=True)
    log(f"Predicting {cons.TARGET_COLUMN}", args.verbose)

    predictions = get_predictions(model, df)
    save_predictions(predictions, args.output_path, args.verbose)
    
    probabilities = get_predicted_probabilities(model, df)
    save_predicted_probabilities(probabilities, args.output_path, args.verbose)

if __name__ == '__main__':
    main()
