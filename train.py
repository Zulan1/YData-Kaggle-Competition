import numpy as np
import optuna
import pickle
import os
import time
import wandb
import xgboost as xgb
import lightgbm as lgb

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris #placehold until preprocessed data is available
from sklearn.model_selection import train_test_split

from app.metrics import compute_score
from app.argparser import get_train_args
from app.helper_functions import load_training_data, split_dataset_Xy

models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
}

def get_model(args, X_train, y_train, X_val, y_val):
    if args.optuna_search:
        best_params = hyperparameter_search(X_train, y_train, X_val, y_val, args)
        model_type = best_params['model_type']
        del best_params['model_type']
        model = models[model_type]
        model.set_params(**best_params)
        return model

    model_type = args.model_type
    match model_type:
        case 'LogisticRegression':
            model = LogisticRegression(C=args.C)
        case 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                criterion=args.criterion,
                )
        case 'SVM':
            model = SVC(
                C=args.C,
                kernel=args.kernel,
                )
        case 'XGBoost':
            model = xgb.XGBClassifier(
                eta=args.eta,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                subsample=args.subsample,
                gamma=args.gamma,
                reg_lambda=args.reg_lambda,
            )
        case 'LightGBM':
            model = lgb.LGBMClassifier(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
            )
        case _:
            raise ValueError('Invalid model type')

    return model


def hyperparameter_search(X_train, y_train, X_val, y_val, args):
    best_score = 0

    def log_score(study, trial):
        nonlocal best_score
        if trial.value > best_score:
            best_score = trial.value
        wandb.log({'best_score': best_score})

    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['LogisticRegression', 'RandomForest', 'SVM'])
        match model_type:
            case 'LogisticRegression':
                hparams = {
                    'C': trial.suggest_float('C', 1e-10, 1e10, log=True)
                }
                model = LogisticRegression()
            case 'RandomForest':
                hparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
                }
                model = RandomForestClassifier()
            case 'SVM':
                hparams = {
                    'C': trial.suggest_float('C', 1e-10, 1e10, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                }
                model = SVC()
            case 'XGBoost':
                hparams = {
                    'eta': trial.suggest_float('eta', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    }
                model = xgb.XGBClassifier()
            case 'LightGBM':
                hparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
                }
                model = lgb.LGBMClassifier()
            case _:
                raise ValueError('Invalid model type')

        model.set_params(**hparams)
        model.fit(X_train, y_train)
        score = compute_score(args.scoring_method, model.predict(X_val), y_val)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_score])
    print(f"Finished {args.n_trials} found best params: {study.best_params}.")
    return study.best_params

def main():
    args = get_train_args()

    #Load processed data:
    # folds, train_set, test_set = load_training_data()
    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    wandb.init(
        project='ydata-kaggle-competition',
        config=args,
        )

    model = get_model(args, X_train, y_train, X_val, y_val)
    
    os.makedirs('models', exist_ok=True)
    timestamp = time.time()
    with open(f'models/model_{timestamp:.0f}.pkl', 'wb') as p:
        pickle.dump(model, p)   
            
if __name__ == '__main__':
    main()
