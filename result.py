import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import constants as cons
from sklearn.metrics import f1_score, roc_auc_score
import os

def main():
    args = get_result_args()
    predictions = pd.read_csv(args.predictions_path)
    labels = pd.read_csv(args.labels_path)
    features = pd.read_csv(args.features_path)
    print("F1 Score: ", f1_score(labels, predictions))
    print("AUC Score: ", roc_auc_score(labels, predictions))

if __name__ == '__main__':
    main()

