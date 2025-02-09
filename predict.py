from app.argparser import get_predict_args
import pandas as pd
import pickle
import constants as cons
import os
from app.helper_functions import log
from app.file_manager import get_model, get_data, save_predictions

def main():
    args = get_predict_args()
    verbose = args.verbose
    model = get_model(args.model_path, verbose)
    df = get_data(args.input_path, verbose)
    os.makedirs(args.output_path, exist_ok=True)

    log(f"Predicting {cons.TARGET_COLUMN}", verbose)
    predictions = pd.DataFrame(
        model.predict(df), 
        index=df.index,
        columns=[cons.TARGET_COLUMN]
    )

    save_predictions(predictions, args.output_path, verbose)

if __name__ == '__main__':
    main()
