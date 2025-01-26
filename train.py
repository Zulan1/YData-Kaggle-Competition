import optuna
import pickle
import os
import time
import wandb
import xgboost as xgb
import lightgbm as lgb
import constants as cons



from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_diabetes

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
        wandb.log({f'best {args.scoring_method}': best_score})

    def objective(trial):
        model_types = ['LogisticRegression', 'RandomForest', 'XGBoost'] if args.model_type is None else [args.model_type]
        model_type = trial.suggest_categorical('model_type', model_types)
        match model_type:
            case 'LogisticRegression':
                hparams = {
                    'C': trial.suggest_float('C', 1e-10, 1e10, log=True),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
                }
                model = LogisticRegression()
            case 'RandomForest':
                hparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 20, 100),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
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
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                }
                model = lgb.LGBMClassifier()
            case _:
                raise ValueError('Invalid model type')

        print(f"\n\nStarted trial {trial.number} with params: {trial.params}.")
        model.set_params(**hparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"{(y_pred == 1).sum()} out of {len(y_pred)} are 1")
        score = compute_score(args.scoring_method, y_val, y_pred)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_score])
    print(f"Finished {args.n_trials} found best params: {study.best_params}, with score: {study.best_value}.")
    return study.best_params

def main():
    args = get_train_args()

    df_train, df_val, _ = load_training_data()
    X_train, y_train = split_dataset_Xy(df_train)
    X_val, y_val = split_dataset_Xy(df_val)
    selected_columns = cons.DEMOGRAPHICS
    selected_columns = [col for col in X_train.columns if any(c in col for c in selected_columns)]

    # X_train.drop(columns=['DateTime', 'user_id', 'session_id'], inplace=True)
    # X_val.drop(columns=['DateTime', 'user_id', 'session_id'], inplace=True)\
    # X_train = X_train[selected_columns]
    # X_val = X_val[selected_columns]
    # X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    

    dmy_cls = DummyClassifier(strategy='most_frequent')
    dmy_cls.fit(X_train, y_train)
    baseline_score = compute_score(args.scoring_method, dmy_cls.predict(X_val), y_val)
    c_mat = confusion_matrix(y_val, dmy_cls.predict(X_val))
    print(f"Baseline confusion matrix:\n{c_mat}")
    print(f"Baseline score: {baseline_score}\n\n")

    wandb.init(
        project='ydata-kaggle-competition',
        config=args,
        )

    model = get_model(args, X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)
    score = compute_score(args.scoring_method, y_val, model.predict(X_val))
    wandb.log({args.scoring_method: score})
    print(f"Final score: {score}")
    
    os.makedirs('models', exist_ok=True)
    timestamp = time.time()
    with open(f'models/model_{timestamp:.0f}.pkl', 'wb') as p:
        pickle.dump(model, p)   
            
if __name__ == '__main__':
    main()
