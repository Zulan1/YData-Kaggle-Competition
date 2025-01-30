import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import constants as cons
import os

def get_predictions(predictions_path):
    return pd.read_csv(predictions_path)

def get_labels(labels_path):
    return pd.read_csv(labels_path)

def main():
    args = get_result_args()
    run_id = args.run_id
    predictions_path = os.path.join(args.input_path, f"predict_{run_id}/{cons.DEFAULT_PREDICTIONS_FILE}")
    predictions = pd.read_csv(predictions_path)

    features_path = os.path.join(args.input_path, f"preprocess_{run_id}/{cons.DEFAULT_PROCESSED_TEST_FILE}")
    features = pd.read_csv(features_path)

    df = pd.concat([features, predictions], axis=1)
    output_path = os.path.join(args.output_path, f"result_{run_id}")
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{cons.DEFAULT_RESULTS_FILE}")
    df.to_csv(file_path, index=False)

    if args.verbose:
        print(f"Results saved to {file_path}")

if __name__ == '__main__':
    main()

