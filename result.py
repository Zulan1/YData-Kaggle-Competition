import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
from app.file_manager import get_model
import os

def main():
    args = get_result_args()
   # print(f"\nScores:")
   # print(f"F1 Score: {f1_value:.4f}")
   # print(f"AUC Score: {auc_value:.4f}")

if __name__ == '__main__':
    main()

