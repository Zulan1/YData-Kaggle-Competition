import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score, confusion_matrix, classification_report
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
    f1_value = f1_score(labels, predictions)
    auc_value = roc_auc_score(labels, predicted_probabilities)
    print(f"F1 Score: {f1_value:.4f}")
    print(f"AUC Score: {auc_value:.4f}")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Match importance scores with feature names from the features DataFrame.
        if len(importances) == len(features.columns):
            feature_names = features.columns
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:")
        for _, row in fi_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
    else:
        print("\nModel does not have a feature_importances_ attribute.")

if __name__ == '__main__':
    main()

