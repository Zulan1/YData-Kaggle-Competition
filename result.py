import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import constants as cons
import os

def main():
    args = get_result_args()
    run_id = args.run_id
    predictions_path = os.path.join(args.input_path, f"predict_{run_id}/{cons.DEFAULT_PREDICTIONS_FILE}")
    predictions = pd.read_csv(predictions_path)

    default_external_test_file = os.path.join(args.input_path, cons.DEFAULT_EXTERNAL_RAW_TEST_FILE)
    features = pd.read_csv(default_external_test_file)

    df = pd.concat([features, predictions], axis=1)
    output_path = os.path.join(args.output_path, f"result_{run_id}")
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{cons.DEFAULT_RESULTS_FILE}")
    df.to_csv(file_path, index=False)

    if args.verbose:
        print(f"Results saved to {file_path}")

if __name__ == '__main__':
    main()

