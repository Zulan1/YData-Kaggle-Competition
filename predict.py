from app.argparser import get_predict_args
import pandas as pd
import pickle
import config as conf
import constants as cons
import os
from app.helper_functions import log
from app.file_manager import get_model, get_test_features, save_predictions
from catboost_transform import catboost_transform

def main():
    args = get_predict_args()
    model = get_model(args.model_path, args.verbose)
    df = get_test_features(args.input_path)
    os.makedirs(args.output_path, exist_ok=True)
    log(f"Predicting {cons.TARGET_COLUMN}", args.verbose)
    predictions = pd.DataFrame(
        model.predict(df),
        index=df.index,
        columns=[cons.TARGET_COLUMN]
    )

    save_predictions(predictions, args.output_path, args.verbose)

if __name__ == '__main__':
    main()
