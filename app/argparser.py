import argparse
import constants as cons

#define training arguments
def get_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--optuna-search', action='store_true', help='Whether to perform hyperparameter search')
    parser.add_argument('--n-trials', type=int, default=100, help='number of trials for hyperparameter search')
    parser.add_argument('--scoring-method', type=str, default='f1', help='The metric to use for evaluation')
    parser.add_argument('--model-type', type=str, default=None, help='The type of model to train')

    parser.add_argument('--C', type=float, default=None, help='The regularization strength for Logistic Regression or SVM')

    parser.add_argument('--n-estimators', type=int, default=None, help='The number of estimators for Random Forest')
    parser.add_argument('--criterion', type=str, default=None, help='The criterion for Random Forest')

    parser.add_argument('--kernel', type=str, default=None, help='The kernel for SVM')
    parser.add_argument('--run-id', type=str, help='Run ID')
    parser.add_argument('--output-path', type=str, default='models', help='Path to the trained model')

    return parser.parse_args()

def get_preprocessing_args():
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--input-path', default=cons.DEFAULT_RAW_TRAIN_FILE, type=str, help='CSV filename to load')
    parser.add_argument('--output-path', type=str, help='Output directory of all proccessed csv files: train, val, test')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--test', action='store_true', help='Preprocess external test set only')
    parser.add_argument('--run-id', type=str, help='Run ID')
    return parser.parse_args()


def get_predict_args():
    parser = argparse.ArgumentParser(description='Predict on new data')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data for prediction')
    parser.add_argument('--output-path', type=str, help='Path to output data for prediction')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--run-id', type=str, help='Run ID')
    return parser.parse_args()

def get_result_args():
    parser = argparse.ArgumentParser(description='Analyze results')
    parser.add_argument('--input-path', type=str, required=True, help='Path to predictions file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to results file')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--run-id', type=str, help='Run ID')
    return parser.parse_args()