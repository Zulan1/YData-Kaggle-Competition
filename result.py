import argparse
from app.argparser import get_result_args
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import constants as cons
import os

def main():
    args = get_result_args()
    run_id = args.run_id
    predictions_path = os.path.join(args.input_path, f"predict_{run_id}/{cons.DEFAULT_PREDICTIONS_FILE}")
    predictions = pd.read_csv(predictions_path)

    default_internal_test_file = os.path.join(args.input_path, f"preprocess_{run_id}/{cons.DEFAULT_TEST_SET_FILE}")
    features = pd.read_csv(default_internal_test_file)

    df = pd.concat([features, predictions], axis=1)
    output_path = os.path.join(args.output_path, f"result_{run_id}")
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{cons.DEFAULT_RESULTS_FILE}")
    df.to_csv(file_path, index=False)

    if args.verbose:
        print(f"Results saved to {file_path}")
    if args.error_analysis:
        labels = pd.read_csv(os.path.join(args.input_path, f"preprocess_{run_id}/{cons.DEFAULT_HOLDOUT_LABELS_FILE}"))
        print("F1 score: ", f1_score(labels, predictions))
        print("Confusion matrix: ", confusion_matrix(labels, predictions))
        print("Classification report: ", classification_report(labels, predictions))
        # For error analysis, create a DataFrame with actual vs predicted
        analysis_df = pd.concat([features, labels, predictions], axis=1)
        analysis_df.columns = list(features.columns) + ['actual', 'predicted']
        
        # Find incorrect predictions
        analysis_df['incorrect'] = analysis_df['actual'] != analysis_df['predicted']

        # Save error analysis to file
        analysis_df.to_csv(os.path.join(output_path, "data_for_error_analysis.csv"), index=False)
        
        # Group by each feature and calculate error rate
        feature_errors = {}
        for column in features.columns:
            if column not in cons.INDEX_COLUMNS:  # Skip ID columns
                errors = analysis_df.groupby(column)['incorrect'].mean().sort_values(ascending=False)
                feature_errors[column] = errors
                
                if args.verbose:
                    print(f"\nError rates by {column}:")
                    print(errors)
                    print(f"Sample size for each {column} value:")
                    print(analysis_df.groupby(column).size())
        
        # Save detailed error analysis to file
        error_analysis_path = os.path.join(output_path, "error_analysis.txt")
        with open(error_analysis_path, 'w') as f:
            f.write("ERROR ANALYSIS BY FEATURE\n")
            f.write("=========================\n\n")
            for feature, errors in feature_errors.items():
                f.write(f"\nError rates by {feature}:\n")
                f.write(str(errors))
                f.write(f"\n\nSample size for each {feature} value:\n")
                f.write(str(analysis_df.groupby(feature).size()))
                f.write("\n" + "="*50 + "\n")
        
        if args.verbose:
            print(f"\nDetailed error analysis saved to {error_analysis_path}")


if __name__ == '__main__':
    main()

