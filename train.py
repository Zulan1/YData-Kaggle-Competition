import optuna
import pickle
import os
import wandb
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import cupy as cp
import pandas as pd
import constants as cons
from app.helper_functions import get_transformer

import pandas as pd


from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


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
                criterion=args.criterion,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                class_weight=args.class_weight,
            )

        case 'XGBoost':
            model = xgb.XGBClassifier(
                eta=args.eta,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                subsample=args.subsample,
                gamma=args.gamma,
                reg_lambda=args.reg_lambda,
                device='cuda' if cp.cuda.is_available() else 'cpu',
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
                device='gpu' if cp.cuda.is_available() else 'cpu',
            )
        case 'CatBoost':
            model = cb.CatBoostClassifier(
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                task_type='GPU' if cp.cuda.is_available() else 'CPU',
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
        model_types = ['XGBoost', 'LightGBM', 'CatBoost'] if args.model_type is None else [args.model_type]
        model_type = trial.suggest_categorical('model_type', model_types)
        match model_type:
            case 'DecisionTree':
                hparams = {
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                    }
                model = DecisionTreeClassifier()
            case 'XGBoost':
                hparams = {
                    'eta': trial.suggest_float('eta', 0.001, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 10, 5000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'device': 'cuda' if cp.cuda.is_available() else 'cpu'
                    }
                model = xgb.XGBClassifier()
            case 'LightGBM':
                hparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 16),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'device': 'gpu' if cp.cuda.is_available() else 'cpu',
                    'verbose': -1,
                    }
                model = lgb.LGBMClassifier()
            case 'CatBoost':
                hparams = {
                    'iterations': trial.suggest_int('iterations', 100, 5000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 10),
                    'task_type': 'GPU' if cp.cuda.is_available() else 'CPU',
                    'verbose': 500,
                    }
                model = cb.CatBoostClassifier()
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
    os.makedirs('plots', exist_ok=True)
    # optuna.visualization.plot_optimization_history(study).write_image('plots/optuna_history.png')
    # optuna.visualization.plot_param_importances(study).write_image('plots/optuna_importances.png')
    return study.best_params

def enforce_smote(X, transformer):
    # Get categorical columns from transformer
    one_hot_groups = []
    print(X.columns)

    # Group one-hot encoded columns correctly
    for category in cons.COLUMNS_TO_OHE:
        group = [col for col in X.columns if col.startswith(f"{category}_")]
        one_hot_groups.append(group)
    
    print(one_hot_groups)
    
    for col_group in one_hot_groups:
        # For each group of one-hot columns, ensure exactly one 1 per row
        group_data = X[col_group]
        max_cols = group_data.idxmax(axis=1)
        
        # Set all columns in group to 0, then set max column to 1
        X[col_group] = 0
        for row_idx, col in enumerate(max_cols):
            X.at[row_idx, col] = 1
    return X



def main():
    args = get_train_args()

    input_path = args.input_path
    run_id = args.run_id
    transformer = get_transformer(input_path)


    
    train_path = os.path.join(input_path, cons.DEFAULT_TRAIN_SET_FILE)
    val_path = os.path.join(input_path, cons.DEFAULT_VAL_SET_FILE)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    X_train, y_train = split_dataset_Xy(df_train)
    X_val, y_val = split_dataset_Xy(df_val)
    # selected_columns = cons.DEMOGRAPHICS
    # selected_columns = [col for col in X_train.columns if any(c in col for c in selected_columns)]

    # X_train.drop(columns=['DateTime', 'user_id', 'session_id'], inplace=True)
    # X_val.drop(columns=['DateTime', 'user_id', 'session_id'], inplace=True)
    # X_train = X_train[selected_columns]
    # X_val = X_val[selected_columns]
   # X_train, y_train = SMOTE().fit_resample(X_train, y_train)
   # X_train = enforce_smote(X_train, transformer)


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
    
    # predictions = model.predict(X_test)
    # score = compute_score(args.scoring_method, y_test, predictions)
    # print(confusion_matrix(y_test, predictions))
    # print(f"Final Test Model score {args.scoring_method}: {score}")
    # wandb.log({f'test_{args.scoring_method}': score})

            
if __name__ == '__main__':
    main()
