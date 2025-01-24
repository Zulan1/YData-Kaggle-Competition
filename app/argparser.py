import argparse
import constants as cons

#define training arguments
def get_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--optuna-search', action='store_true', help='Whether to perform hyperparameter search')
    parser.add_argument('--n-trials', type=int, default=100, help='number of trials for hyperparameter search')
    parser.add_argument('--trial-epochs', type=int, default=100, help='Number of epochs per trial for hyperparameter search')
    parser.add_argument('--scoring-method', type=str, default='f1', help='The metric to use for evaluation')
    parser.add_argument('--model-type', type=str, default='RandomForest', help='The type of model to train')

    parser.add_argument('--C', type=float, default=0, help='The regularization strength for Logistic Regression or SVM')

    parser.add_argument('--n-estimators', type=int, default=100, help='The number of estimators for Random Forest')
    parser.add_argument('--criterion', type=str, default='gini', help='The criterion for Random Forest')

    parser.add_argument('--kernel', type=str, default='rbf', help='The kernel for SVM')

    return parser.parse_args()

def get_preprocessing_args():
    parser = argparse.ArgumentParser(description='Preprocess the data')

    parser.add_argument('--input-path', default=cons.DEFAULT_RAW_TRAIN_FILE, type=str, help='CSV filename to load')
    parser.add_argument('--out-path', type=str, help='Output directory of all proccessed csv files: train, val, test')
    parser.add_argument('--verbose', default=False, type=bool, help='Print additional information')
    parser.add_argument('--test', type=bool, help='Run on test rather on train dataset')

    return parser.parse_args()
