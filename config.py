USE_CATBOOST = True
IMPUTATION_STATEGY = 'most_frequent'
RANDOM_STATE = 42

LEAVE_ONE_PRODUCT_OUT = True

DEFAULT_MODEL = 'CatBoost'
CAT_FEATURES = ['product', 'campaign_id', 'webpage_id', 'product_category', 'user_group_id', 'gender', 'var_1', 'session_time_of_day', 'user_depth'] + \
['secondary_product_category_known', 'city_development_index_known', 'session_day_of_week']


DEFAULT_MODEL_PARAMS = {
    "iterations": 1500,  # Number of boosting rounds
    "learning_rate": 0.05,  # Lower = better generalization
    "depth": 7,  # Balanced depth to prevent overfitting
    "l2_leaf_reg": 10,  # Regularization to control complexity
    "loss_function": "Logloss",  # Best for CTR probability modeling
    "eval_metric": "AUC",  # Optimized for ranking CTR probabilities
    "cat_features": CAT_FEATURES,  # 🚀 No encoding needed!
    "bagging_temperature": 1,  # Adds randomness for generalization
    "random_strength": 2,  # Adds noise to prevent overfitting
    "boosting_type": "Ordered",  # "Plain" is faster, "Ordered" is better for small datasets
    "subsample": 0.8,  # Helps prevent overfitting (sample 80% of data per iteration)
    "task_type": "CPU",  # 🚀 Run on local machine (CPU)
    "verbose": 500
}
