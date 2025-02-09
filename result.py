import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import constants as cons
import os
from app.file_manager import get_data

def main():
    args = get_result_args()
    predictions = get_data(args.predictions_path, args.verbose)
    labels = get_data(args.labels_path, args.verbose)
    features = get_data(args.features_path, args.verbose)
    print("F1 Score: ", f1_score(labels, predictions))
    print("Confusion Matrix: ", confusion_matrix(labels, predictions))
if __name__ == '__main__':
    main()

