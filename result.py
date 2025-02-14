import argparse
from app.argparser import get_result_args
from app.metrics import compute_metrics, print_metrics
import pandas as pd
import constants as cons
from sklearn.metrics import f1_score, roc_auc_score
from app.file_manager import get_model
import os

def main():
    args = get_result_args()
    model = get_model(args.model_path, args.verbose)
    features = pd.read_csv(args.features_path)
    predictions = pd.read_csv(args.predictions_path)
    labels = pd.read_csv(args.labels_path)
    predicted_probabilities = pd.read_csv(args.predicted_probabilities_path)
    
    metrics = compute_metrics(labels, predictions, predicted_probabilities)

    print_metrics(metrics)


if __name__ == '__main__':
    main()

