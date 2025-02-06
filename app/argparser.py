import argparse
import constants as cons

#define training arguments
def get_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data for training')
    parser.add_argument('--optuna-search', action='store_true', help='Whether to perform hyperparameter search')
    parser.add_argument('--n-trials', type=int, default=100, help='number of trials for hyperparameter search')
    parser.add_argument('--scoring-method', type=str, default='f1', help='The metric to use for evaluation')
    parser.add_argument('--model-type', type=str, default=None, help='The type of model to train')
    parser.add_argument('--eta', type=float, default=None, help='The learning rate for XGBoost')
    parser.add_argument('--n-estimators', type=int, default=None, help='The number of estimators for XGBoost')
    parser.add_argument('--max-depth', type=int, default=None, help='The maximum depth for XGBoost')
    parser.add_argument('--subsample', type=float, default=None, help='The fraction of samples for XGBoost')
    parser.add_argument('--gamma', type=float, default=None, help='The minimum loss reduction required to make a split for XGBoost')
    parser.add_argument('--reg-lambda', type=float, default=None, help='The L2 regularization term on weights for XGBoost')
    parser.add_argument('--scale-pos-weight', type=float, default=None, help='The scale_pos_weight parameter for XGBoost')

    parser.add_argument('--C', type=float, default=None, help='The regularization strength for Logistic Regression or SVM')

    parser.add_argument('--criterion', type=str, default=None, help='The criterion for Random Forest')
    parser.add_argument('--min-samples-split', type=int, default=None, help='The minimum number of samples required to split an internal node for Random Forest')
    parser.add_argument('--class-weight', type=str, default=None, help='The class weight for Random Forest')

    parser.add_argument('--kernel', type=str, default=None, help='The kernel for SVM')
    parser.add_argument('--run-id', type=str, help='Run ID')
    parser.add_argument('--output-path', type=str, default='models', help='Path to the trained model')

    return parser.parse_args()

def get_preprocessing_args():
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--input-path', type=str, help='Folder with input data')
    parser.add_argument('--output-path', type=str, help='Output directory of all proccessed csv files: train, val, test')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train, test')
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
    parser.add_argument('--error-analysis', action='store_true', help='Analyze test set')
    return parser.parse_args()