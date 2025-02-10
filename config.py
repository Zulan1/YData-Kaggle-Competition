import constants as cons

USE_CATBOOST = True
TRAIN_TEST_VAL_SPLIT = (0.6, 0.2, 0.2)
KNN_IMPUTATION_NNEIGHBORS = 5
IMPUTATION_STATEGY = 'most_frequent'
RANDOM_STATE = 42
LEAVE_ONE_OUT = True

DEFAULT_MODEL = 'CatBoost'
CAT_FEATURES = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'var_1', 'user_depth'] + \
[f'most_common_{col}' for col in ['product', 'product_category_1', 'campaign_id', 'webpage_id']]

DEFAULT_MODEL_PARAMS = {
    "iterations": 1000,  # Number of boosting rounds
    "learning_rate": 0.05,  # Lower = better generalization
    "depth": 6,  # Balanced depth to prevent overfitting
    "l2_leaf_reg": 5,  # Regularization to control complexity
    "loss_function": "Logloss",  # Best for CTR probability modeling
    "eval_metric": "AUC",  # Optimized for ranking CTR probabilities
    "cat_features": CAT_FEATURES,  # 🚀 No encoding needed!
    "scale_pos_weight": 13,  # 7% CTR
    "bagging_temperature": 1,  # Adds randomness for generalization
    "random_strength": 1,  # Adds noise to prevent overfitting
    "boosting_type": "Ordered",  # "Plain" is faster, "Ordered" is better for small datasets
    "subsample": 0.8,  # Helps prevent overfitting (sample 80% of data per iteration)
    "task_type": "CPU",  # 🚀 Run on local machine (CPU)
    "verbose": 100  # Print progress every 100 iterations
}