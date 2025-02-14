import numpy as np
from sklearn.preprocessing import OneHotEncoder
from feature_engineering import NUMERICAL_THRESHOLD_FEATURES, CATEGORICAL_THRESHOLD_FEATURES
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
"""
This is a meta-model that uses the predicted probabilities from the base model to adjust the threshold for the final prediction.
"""

META_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": None,
    "n_jobs": -1,
    "bootstrap": True,
    "class_weight": 'balanced_subsample',
}

class MetaDataTransformer:
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.categorical_threshold_features = CATEGORICAL_THRESHOLD_FEATURES
        self.numerical_threshold_features = NUMERICAL_THRESHOLD_FEATURES
    def fit(self, X):
        X_categorical = X[self.categorical_threshold_features]
        self.ohe.fit(X_categorical)
    def transform(self, X, y_pred_prob):
        X_categorical = X[self.categorical_threshold_features]
        X_categorical = self.ohe.transform(X_categorical)
        ohe_columns = self.ohe.get_feature_names_out()
        return pd.DataFrame(np.column_stack([y_pred_prob, X[self.numerical_threshold_features], X_categorical]), columns=["y_pred_prob", *self.numerical_threshold_features, *ohe_columns])

class MetaModel:
    def __init__(self, base_model):
        self.base_model = base_model  # CatBoost model
        self.meta_model = RandomForestClassifier(**META_MODEL_PARAMS)
        self.meta_data_transformer = MetaDataTransformer()

    def fit(self, X_train, y_train):
        # Train base model
        self.base_model.fit(X_train, y_train)
        print("Base model trained")
        self.meta_data_transformer.fit(X_train)
        print("Meta data transformer fitted")
        
        # Get predicted probabilities
        y_pred_prob_train = self.base_model.predict_proba(X_train)[:, 1]
        
        # Prepare meta features (predicted probabilities + session/user-level features)
        X_meta_train = self.meta_data_transformer.transform(X_train, y_pred_prob_train)
        print("Meta features transformed", X_meta_train.columns)
        
        # Train meta-model
        self.meta_model.fit(X_meta_train, y_train)
        print("Meta model trained")
    def predict(self, X):
        # Get predicted probabilities from base model
        y_pred_prob = self.base_model.predict_proba(X)[:, 1]
        X_meta = self.meta_data_transformer.transform(X, y_pred_prob)
        # Predict final labels using meta-model
        return self.meta_model.predict(X_meta)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)[:, 1]  # Keep probabilities accessible
