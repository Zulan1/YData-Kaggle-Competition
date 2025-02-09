from app.argparser import get_predict_args
import pandas as pd
import pickle
import constants as cons
import os
from app.helper_functions import log


def get_data(input_path: str, run_id: str) -> pd.DataFrame:
    """Load preprocessed test data."""
    data_path = os.path.join(input_path, f"preprocess_{run_id}/{cons.DEFAULT_TEST_SET_FILE}")
    return pd.read_csv(data_path)

def main():
    args = get_predict_args()
    run_id = args.run_id
    verbose = args.verbose

    log("Loading model", verbose)
    with open(args.model_path, 'rb') as p:
        model = pickle.load(p)
    log(f"Model type: {model.__class__.__name__}", verbose)
    
    log(f"Loading data from {args.input_path}", verbose)
    df = get_data(args.input_path, run_id)

    log(f"Predicting {cons.TARGET_COLUMN}", verbose)
    predictions = pd.DataFrame(
        model.predict(df), 
        index=df.index,
        columns=[cons.TARGET_COLUMN]
    )

    # Save predictions
    output_dir = os.path.join(args.output_path, f"predict_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cons.DEFAULT_PREDICTIONS_FILE)
    predictions.to_csv(output_path, index=False)
    log(f"Predictions saved to {output_path}", verbose)

if __name__ == '__main__':
    main()
