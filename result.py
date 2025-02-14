import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
from app.file_manager import get_model
import os

def main():
    args = get_result_args()
    model = get_model(args.model_path, args.verbose)
    features = pd.read_csv(args.features_path)
    predictions = pd.read_csv(args.predictions_path)
    labels = pd.read_csv(args.labels_path)
    predicted_probabilities = pd.read_csv(args.predicted_probabilities_path)
    auc_value = roc_auc_score(labels, predicted_probabilities)
    print(f"[result.py] ROC AUC Score: {auc_value:.4f}")
   # print(f"\nScores:")
   # print(f"F1 Score: {f1_value:.4f}")
   # print(f"AUC Score: {auc_value:.4f}")

if __name__ == '__main__':
    main()

