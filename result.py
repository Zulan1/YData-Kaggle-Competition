import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import constants as cons

def get_predictions(predictions_path):
    return pd.read_csv(predictions_path)

def get_labels(labels_path):
    return pd.read_csv(labels_path)

def main():
    args = get_result_args()

    if args.verbose:
        print(f"Loading predictions from {args.predictions_path}")

    df = get_predictions(args.predictions_path)

    if args.verbose:
        print(f"Loading labels from {args.labels_path}")

    labels = get_labels(args.labels_path)

    print(f"Analyzing results from {args.predictions_path}")
    print(f"F1 score: {f1_score(labels, df[cons.TARGET])}")
    print(f"Confusion matrix: {confusion_matrix(labels, df[cons.TARGET])}")

    if args.verbose:
        print(f"Saving results to {args.results_path}")

    df.to_csv(args.results_path, index=False)


if __name__ == '__main__':
    main()
