import optuna
import pickle
import os
import wandb
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pandas as pd
import constants as cons
from app.helper_functions import get_transformer

import pandas as pd


from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures


from app.metrics import compute_score
from app.argparser import get_train_args
from app.helper_functions import split_dataset_Xy

models = {
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'CatBoost': cb.CatBoostClassifier(),
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
        case 'DecisionTree':
            model = DecisionTreeClassifier(
                criterion = args.criterion,
                max_depth = args.max_depth,
                min_samples_split = args.min_samples_split,
                class_weight = 'balanced' if args.class_weight else None,
            )

        case 'XGBoost':
            model = xgb.XGBClassifier(
                eta = args.eta,
                n_estimators = args.n_estimators,
                max_depth = args.max_depth,
                subsample = args.subsample,
                gamma = args.gamma,
                reg_lambda = args.reg_lambda,
                scale_pos_weight =  y_train.value_counts()[0] / y_train.value_counts()[1] if args.scale_pos_weight else 1,
                device = 'cuda' if args.gpu else 'cpu',
            )

        case 'LightGBM':
            model = lgb.LGBMClassifier(
                n_estimators = args.n_estimators,
                learning_rate = args.learning_rate,
                max_depth = args.max_depth,
                subsample = args.subsample,
                colsample_bytree = args.colsample_bytree,
                reg_alpha = args.reg_alpha,
                reg_lambda = args.reg_lambda,
                is_balanced =  args.is_balanced,
                device = 'gpu' if args.gpu else 'cpu',
            )
        case 'CatBoost':
            model = cb.CatBoostClassifier(
                iterations = args.iterations,
                learning_rate = args.learning_rate,
                depth = args.depth,
                l2_leaf_reg = args.l2_leaf_reg,
                auto_class_weights = 'balanced' if args.class_weights else None,
                task_type='GPU' if args.gpu else 'CPU',
            )
        case _:
            raise ValueError('Invalid model type')

    return model


def hyperparameter_search(X_train, y_train, X_val, y_val, args):

    def log_score(study, trial):
        val_score, val_precision, val_recall, train_score = study.best_trials[0].values
        log_msg = {
            f'best val {args.scoring_method}': val_score,
            'best val precision': val_precision,
            'best val recall': val_recall,
            f'best train {args.scoring_method}': train_score,
        }
        wandb.log(log_msg)

    def objective(trial):
        model_types = ['XGBoost', 'LightGBM', 'CatBoost'] if args.model_type is None else [args.model_type]
        model_type = trial.suggest_categorical('model_type', model_types)
        match model_type:
            case 'DecisionTree':
                hparams = {
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
                    }
                model = DecisionTreeClassifier()
            case 'XGBoost':
                hparams = {
                    'eta': trial.suggest_float('eta', 0.001, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 10, 5000, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [y_train.value_counts()[0] / y_train.value_counts()[1], 1]),
                    'device': 'cuda' if args.gpu else 'cpu'
                    }
                model = xgb.XGBClassifier()
            case 'LightGBM':
                hparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 5000, log=True),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 16),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'is_balanced': trial.suggest_categorical('is_balanced', [True, False]),
                    'device': 'gpu' if args.gpu else 'cpu',
                    'valid_sets': [(X_val, y_val)],
                    'verbose': -1,
                    }
                model = lgb.LGBMClassifier()
            case 'CatBoost':
                hparams = {
                    'iterations': trial.suggest_int('iterations', 100, 5000, log=True),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 10),
                    'task_type': 'GPU' if args.gpu else 'CPU',
                    'verbose': 500,
                    }
                auto_class_weights = trial.suggest_categorical('auto_class_weights', ['Balanced', None]),
                if auto_class_weights == 'Balanced':
                    hparams['auto_class_weights'] = 'Balanced'
                model = cb.CatBoostClassifier()
            case _:
                raise ValueError('Invalid model type')

        print(f"\n\nStarted trial {trial.number} with params: {trial.params}.")
        model.set_params(**hparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"{(y_pred == 1).sum()} out of {len(y_pred)} are 1")
        train_score = compute_score(args.scoring_method, y_val, y_pred)
        val_score = compute_score(args.scoring_method, y_val, y_pred)
        val_precision = compute_score('precision', y_val, y_pred)
        val_recall = compute_score('recall', y_val, y_pred)

        return val_score, val_precision, val_recall, train_score

    study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize'])
    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_score])
    print(f"Finished {args.n_trials} found best params: {study.best_params}, with score: {study.best_value}.")
    return study.best_trials[0].params


def main():
    args = get_train_args()

    input_path = args.input_path
    
    train_path = os.path.join(input_path, cons.DEFAULT_TRAIN_SET_FILE)
    val_path = os.path.join(input_path, cons.DEFAULT_VAL_SET_FILE)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    X_train, y_train = split_dataset_Xy(df_train)
    X_val, y_val = split_dataset_Xy(df_val)


    strategies = ('most_frequent', 'stratified', 'uniform')
    dmy_scores = []
    for strategy in strategies:
        dmy_cls = DummyClassifier(strategy=strategy)
        dmy_cls.fit(X_train, y_train)
        dmy_scores.append((compute_score(args.scoring_method, dmy_cls.predict(X_val), y_val), dmy_cls))
    baseline_score, dmy_cls = max(dmy_scores)
    c_mat = confusion_matrix(y_val, dmy_cls.predict(X_val))
    print(f"Baseline strategy: {dmy_cls.strategy}")
    print(f"Baseline confusion matrix:\n{c_mat}")
    print(f"Baseline score: {baseline_score}\n\n")

    wandb.init(
        project='ydata-kaggle-competition',
        config=args,
        )
    

    model = get_model(args, X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    score = compute_score(args.scoring_method, y_val, predictions)
    print(confusion_matrix(y_val, predictions))
    wandb.log({args.scoring_method: score})
    print(f"Final score: {score}")

    
    output_path = f'{args.output_path}/train_{args.run_id}'
    os.makedirs(output_path, exist_ok=True)
    with open(f'{output_path}/model.pkl', 'wb') as p:
        pickle.dump(model, p)
    

if __name__ == '__main__':
    main()
