import argparse
import constants as cons

def parse_xgb_params(params):
    keys = ['n_estimators', 'eta', 'max_depth', 'subsample', 'gamma', 'reg_lambda', 'scale_pos_weight']
    if len(params) != len(keys):    
        raise ValueError(f'Expected {len(keys)} hyperparameters, got {len(params)}')
    return {k: params[k] for k in keys}

def parse_lr_params(params):
    return {'C': float(params)}

def parse_lgb_params(params):
    keys = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
    if len(params) != len(keys):    
        raise ValueError(f'Expected {len(keys)} hyperparameters, got {len(params)}')
    return {k: params[k] for k in keys}

def parse_cb_params(params):
    keys = ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg']
    if len(params) != len(keys):    
        raise ValueError(f'Expected {len(keys)} hyperparameters, got {len(params)}')
    return {k: params[k] for k in keys}

def parse_tree_params(params):
    keys = ['criterion', 'max_depth', 'min_samples_split', 'class_weight', 'max_features']
    if len(params) != len(keys):    
        raise ValueError(f'Expected {len(keys)} hyperparameters, got {len(params)}')
    return {k: params[k] for k in keys}

#define training arguments
def get_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data for training')
    parser.add_argument('--optuna-search', action='store_true', help='Whether to perform hyperparameter search')
    parser.add_argument('--n-trials', type=int, default=100, help='number of trials for hyperparameter search')
    parser.add_argument('--scoring-method', type=str, default='auc', help='The metric to use for evaluation')
    parser.add_argument('--model-type', type=str, default=None, help='The type of model to train, or to search on if optuna_search is enabled')
    parser.add_argument('--output-path', type=str, default='models', help='Path to the trained model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    parser.add_argument('--xgb-params', type=parse_xgb_params, default=None,
                        help='XGBoost hyperparameters, required if model-type is XGBoost\n'
                        'Expected keys:\n'
                        'n_estimators - number of trees\n'
                        'eta - learning rate\n'
                        'max_depth - maximum depth of each tree\n'
                        'subsample - fraction of samples used for fitting each tree\n'
                        'gamma - minimum loss reduction required to make a further partition\n'
                        'reg_lambda - L2 regularization term on weights\n'
                        'scale_pos_weight - True/False whether to rebalance classes\n'
                        )

    parser.add_argument('--lgb-params', type=parse_lgb_params, default=None,
                        help='LightGBM hyperparameters, required if model-type is LightGBM\n'
                        'Expected keys:\n'
                        'n_estimators - number of boosting iterations\n'
                        'learning_rate - boosting learning rate\n'
                        'max_depth - maximum tree depth\n'
                        'subsample - fraction of data to be used for each iteration\n'
                        'colsample_bytree - fraction of features to be used for each iteration\n'
                        'reg_alpha - L1 regularization term on weights\n'
                        'reg_lambda - L2 regularization term on weights\n'
                        'is_balanced - True/False whether to rebalance classes\n'
                        )

    parser.add_argument('--cb-params', type=parse_cb_params, default=None,
                        help='CatBoost hyperparameters, required if model-type is CatBoost\n'
                        'Expected keys:\n'
                        'iterations - number of boosting iterations\n'
                        'learning_rate - boosting learning rate\n'
                        'depth - maximum tree depth\n'
                        'l2_leaf_reg - L2 regularization term on leaf weights\n'
                        'class_weights - True/False whether to rebalance classes\n'
                        )

    parser.add_argument('--tree-params', type=parse_tree_params, default=None,
                        help='DecisionTree hyperparameters, required if model-type is DecisionTree\n'
                        'Expected keys:\n'
                        'criterion - function to measure the quality of a split\n'
                        'max_depth - maximum tree depth\n'
                        'min_samples_split - minimum number of samples required to split an internal node\n'
                        'class_weight - True/False whether to rebalance classes\n'
                        )
    parser.add_argument('--lr-params', type=parse_lr_params, default=None,
                        help='LogisticRegression hyperparameters, required if model-type is LogisticRegression\n'
                        'Expected keys:\n'
                        'C - inverse of regularization strength\n'
                        )
    parser.add_argument('--use-default-model', action='store_true', help='Use default model parameters')

    return parser.parse_args()

def get_preprocessing_args():
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--csv-full-path', type=str, help='The full path to the input csv file')
    parser.add_argument('--output-path', type=str, help='Output directory of all proccessed csv files: train, val, test')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train, test')
    parser.add_argument('--transformer-path', type=str, default=None,help='Path to transformer file')
    return parser.parse_args()

def get_predict_args():
    parser = argparse.ArgumentParser(description='Predict on new data')
    parser.add_argument('--csv-for-prediction', type=str, required=True, help='Path to input data for prediction (no labels)')
    parser.add_argument('--model-path', type=str, help='path of the model to use for prediction')
    parser.add_argument('--features-path', type=str, help='path of the features to use for prediction')
    parser.add_argument('--transformer-path', type=str, default=None,help='Path to transformer file')
    parser.add_argument('--output-path', type=str, help='Path to output data for prediction')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    return parser.parse_args()

def get_result_args():
    parser = argparse.ArgumentParser(description='Analyze results')
    parser.add_argument('--predictions-path', type=str, required=True, help="Path to predictions file")
    parser.add_argument('--predicted-probabilities-path', type=str, required=True, help="Path to predicted probabilities file")
    parser.add_argument('--labels-path', type=str, required=True, help="Path to labels file")
    parser.add_argument('--features-path', type=str, required=True, help='Path to features file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to results file')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model file')
    return parser.parse_args()