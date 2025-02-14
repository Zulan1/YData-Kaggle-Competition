DEFAULT_META_MODEL = "RandomForestClassifier"
DEFAULT_META_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": None,
    "n_jobs": -1,
    "bootstrap": True,
    "class_weight": 'balanced_subsample',
}