import optuna
import pickle
import training_constants
import os
import wandb
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pandas as pd
import constants as cons
import json

import pandas as pd
import config as conf

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


from app.metrics import compute_score
from app.file_manager import get_val_set, get_train_set, load_full_processed_training, save_full_model
from app.file_manager import save_model
from app.argparser import get_train_args
from app.helper_functions import split_dataset_Xy
from metamodel import MetaModel

from catboost_transform import catboost_transform

models = {
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'CatBoost': cb.CatBoostClassifier(),
}

def get_model(args, X_train, y_train, X_val, y_val):
    if args.use_default_model:
        model = models[conf.DEFAULT_MODEL]
        model.set_params(**conf.DEFAULT_MODEL_PARAMS)
        return model

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
        best_trial = max(study.best_trials, key=lambda t: t.values[0])
        val_score, val_precision, val_recall, train_score = best_trial.values
        log_msg = {
            f'best val {args.scoring_method}': val_score,
            'best val precision': val_precision,
            'best val recall': val_recall,
            f'best train {args.scoring_method}': train_score,
        }
        print(f"Finished trial {trial.number} with score {trial.values}\n"
              f"Best is trial {best_trial.number} with score {best_trial.values}.")
        wandb.log(log_msg)

    def objective(trial):
        model_types = training_constants.MODELS if args.model_type is None else [args.model_type]
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
                class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
                hparams = {
                    'eta': trial.suggest_float('eta', 0.001, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 10, 5000, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [class_ratio, 1]),
                    'device': 'cuda' if args.gpu else 'cpu',
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
                    'is_unbalanced': trial.suggest_categorical('is_unbalanced', [True, False]),
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
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                    'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
                    'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                    'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 10),
                    'border_count': trial.suggest_int('border_count', 1, 255),
                    'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                    'od_wait': trial.suggest_int('od_wait', 10, 100),
                    'task_type': 'GPU' if args.gpu else 'CPU',
                    'verbose': 500,
                    'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                }
                hparams['auto_class_weights'] = 'Balanced'
                hparams['cat_features'] = conf.CAT_FEATURES
                hparams['valid_sets'] = [(X_val, y_val)]
                hparams['allow_writing_files'] = False
                hparams['random_seed'] = 42
                model = cb.CatBoostClassifier()
            case _:
                raise ValueError('Invalid model type')

        print(f"\n\nStarted trial {trial.number} with params: {trial.params}.")
        model.set_params(**hparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"{(y_pred == 1).sum()} out of {len(y_pred)} are 1")
        y_proba = model.predict_proba(X_val)
        train_score = compute_score(args.scoring_method, y_val, y_pred, y_proba)
        val_score = compute_score(args.scoring_method, y_val, y_pred, y_proba)
        val_precision = compute_score('precision', y_val, y_pred)
        val_recall = compute_score('recall', y_val, y_pred)

        return val_score, val_precision, val_recall, train_score

    study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize'])
    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_score])
    best_trial = max(study.best_trials, key=lambda t: t.values[0])
    print(f"Finished {args.n_trials} found best params: {best_trial.params}, with score: {best_trial.values[0]}.")

    # Log to Weights & Biases (if needed)
    wandb.log({"best_hyperparameters": best_trial.params})

    return best_trial.params

def main():
    # Parse training arguments
    args = get_train_args()

    # Load training and validation data
    df_train = get_train_set(args.input_path)
    df_val = get_val_set(args.input_path)
    X_train, y_train = split_dataset_Xy(df_train)
    X_val, y_val = split_dataset_Xy(df_val)
    if args.verbose:
        print(f"[train.py] Training set loaded. Shape: {X_train.shape}, {y_train.shape}")
        print(f"[train.py] Validation set loaded. Shape: {X_val.shape}, {y_val.shape}")

    # Initialize WandB for tracking
    #wandb.init(project='ydata-kaggle-competition', config=vars(args))
    
    if args.use_default_model:
        if args.verbose:
            print(f"[train.py] Using default CatBoost model hyperparameters")
        # Use default CatBoost hyperparameters from the configuration.
        model_params = conf.DEFAULT_MODEL_PARAMS.copy()  # default dictionary from config
        # Override/add fixed parameters to avoid CatBoost writing files
        model_params.update({
            "cat_features": conf.CAT_FEATURES,
            "task_type": "GPU" if args.gpu else "CPU",
            "verbose": False,
            "snapshot_interval": 0,         # disables snapshot file creation
            "allow_writing_files": False,   # no extra file writing
            "random_seed": 42,
        })
        model = cb.CatBoostClassifier(**model_params)

    else:
        if args.verbose:
            print(f"[train.py] Optimizing CatBoost hyperparameters using Optuna")
        # Optimize CatBoost hyperparameters using Optuna

        def objective(trial):
            hparams = {
                "iterations": trial.suggest_int("iterations", 100, 5000, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
                "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
                "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
                "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 10),
                "border_count": trial.suggest_int("border_count", 1, 255),
                "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
                "od_wait": trial.suggest_int("od_wait", 10, 100),
                "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            }
            # Add fixed parameters
            hparams.update({
                "cat_features": conf.CAT_FEATURES,
                "task_type": "GPU" if args.gpu else "CPU",
                "verbose": False,
                "snapshot_interval": 0,
                "allow_writing_files": False,
                "random_seed": 42,
                "auto_class_weights": "Balanced"
            })
            model = cb.CatBoostClassifier(**hparams)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            predictions = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            score = compute_score(args.scoring_method, y_val, predictions, y_proba)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)
        best_params = study.best_trial.params
        # Add fixed parameters that should not be optimized
        best_params.update({
            "cat_features": conf.CAT_FEATURES,
            "task_type": "GPU" if args.gpu else "CPU",
            "verbose": False,
            "snapshot_interval": 0,
            "allow_writing_files": False,
            "random_seed": 42,
            "auto_class_weights": "Balanced",
        })
        model = cb.CatBoostClassifier(**best_params)

    # Train the final model on training data
    model = MetaModel(model)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)
    print(y_proba.min(), y_proba.max(), y_proba.mean())
    y_pred = model.predict(X_val)
    score = compute_score(args.scoring_method, y_val, y_pred, y_proba)
    print("Final Score on Validation Set: ", score)
    print("F1 Score on Validation Set: ", f1_score(y_val, y_pred))
    print("Balanced Accuracy on Validation Set: ", balanced_accuracy_score(y_val, y_pred))


    
    # Log score and save the model
    #wandb.log({args.scoring_method: score})
    save_model(model, args.output_path)

    if args.verbose:
        print(f"[train.py] Model saved to {args.output_path}.")
    
    print(f"[train.py] Training model on full training set for future inference")

    df_full = load_full_processed_training(args.input_path)
    X_full, y_full = split_dataset_Xy(df_full)
    model.fit(X_full, y_full)
    save_full_model(model, args.output_path)
    if args.verbose:
        print(f"[train.py] Full model saved to {args.output_path}.")

if __name__ == "__main__":
    main()
