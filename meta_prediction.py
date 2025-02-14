import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from feature_engineering import NUMERICAL_THRESHOLD_FEATURES, CATEGORICAL_THRESHOLD_FEATURES
from meta_prediction_constants import DEFAULT_META_MODEL, DEFAULT_META_MODEL_PARAMS
"""
This is a meta-model that uses the predicted probabilities from the base model to adjust the threshold for the final prediction.
"""

def get_meta_model(model_name=DEFAULT_META_MODEL, model_params=DEFAULT_META_MODEL_PARAMS):
    """
    Return an instance of the meta-model specified by model_name using model_params.
    
    Parameters:
        model_name (str): The name of the meta-model.
        model_params (dict): The initialization parameters for the meta-model.
        
    Returns:
        An instance of the meta-model.
        
    Raises:
        ValueError: If the model_name is not recognized.
    """
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"[meta_predictions.py] Unknown meta-model: {model_name}")



class MetaDataTransformer:
    """
    Transformer to create meta-data features from the base model predictions and the original features.
    
    The transformation combines the predicted probabilities, selected numerical features, and
    one-hot-encoded categorical features.
    """
    def __init__(self, verbose=False):
        """
        Initialize the transformer with the list of features and a OneHotEncoder.
        
        Parameters:
            verbose (bool): If True, prints verbose output.
        """
        self.verbose = verbose
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.categorical_threshold_features = CATEGORICAL_THRESHOLD_FEATURES
        self.numerical_threshold_features = NUMERICAL_THRESHOLD_FEATURES

    def fit(self, X):
        """
        Fit the OneHotEncoder on the categorical threshold features.
        
        Parameters:
            X (pandas.DataFrame): The training data.
        """
        if self.verbose:
            print("[meta_predictions.py] Fitting OneHotEncoder on categorical features.")
        X_categorical = X[self.categorical_threshold_features]
        self.ohe.fit(X_categorical)

    def transform(self, X, y_pred_prob):
        """
        Transform the data to generate meta-features.
        
        Combines the base model's predicted probabilities, numerical features,
        and one-hot-encoded categorical features into a single DataFrame.
        
        Parameters:
            X (pandas.DataFrame): Data to transform.
            y_pred_prob (array-like): Base model predicted probabilities.
            
        Returns:
            pandas.DataFrame: A DataFrame containing the meta-data features.
        """
        if self.verbose:
            print("[meta_predictions.py] Transforming data to generate meta features.")
        X_categorical_transformed = self.ohe.transform(X[self.categorical_threshold_features])
        ohe_columns = self.ohe.get_feature_names_out()
        # Combine predicted probabilities, numerical features and transformed categorical features
        combined_data = np.column_stack([y_pred_prob, X[self.numerical_threshold_features], X_categorical_transformed])
        column_names = ["y_pred_prob", *self.numerical_threshold_features, *ohe_columns]
        return pd.DataFrame(combined_data, columns=column_names)

class MetaModel:
    """
    A meta-model that adjusts the final prediction thresholds using the predicted probabilities
    from a base model combined with additional engineered features.
    """
    def __init__(self, base_model, verbose=False):
        """
        Initialize the MetaModel.
        
        Parameters:
            base_model: An instance of the base model. It must implement the fit() and predict_proba() methods.
            verbose (bool): If True, prints verbose output.
        """
        self.verbose = verbose
        self.base_model = base_model
        # Initialize the meta-model (using defaults from constants)
        self.meta_model = get_meta_model()
        self.meta_data_transformer = MetaDataTransformer(verbose=self.verbose)

    def fit(self, X_train, y_train):
        """
        Fit both the base model and the meta-model.
        
        Steps:
            1. Fit the base model.
            2. Fit the meta data transformer.
            3. Generate meta features using base model probabilities.
            4. Fit the meta-model on these features.
        
        Parameters:
            X_train (pandas.DataFrame): The training features.
            y_train (array-like): The training targets.
        """
        if self.verbose:
            print("[meta_predictions.py] Fitting base model.")
        # Train the base model
        self.base_model.fit(X_train, y_train)
        
        if self.verbose:
            print("[meta_predictions.py] Fitting meta data transformer.")
        self.meta_data_transformer.fit(X_train)
        
        if self.verbose:
            print("[meta_predictions.py] Generating meta features from base model predictions.")
        # Get predicted probabilities from the base model for training
        y_pred_prob_train = self.base_model.predict_proba(X_train)[:, 1]
        X_meta_train = self.meta_data_transformer.transform(X_train, y_pred_prob_train)
        
        if self.verbose:
            print("[meta_predictions.py] Fitting meta model.")
        # Train the meta-model with the enriched features
        self.meta_model.fit(X_meta_train, y_train)

    def predict(self, X):
        """
        Predict the final output using the meta-model.
        
        Parameters:
            X (pandas.DataFrame): The input features.
            
        Returns:
            array-like: The predicted labels.
        """
        if self.verbose:
            print("[meta_predictions.py] Generating predictions using the meta model.")
        # Generate predicted probabilities from the base model
        y_pred_prob = self.base_model.predict_proba(X)[:, 1]
        # Construct meta features from the predicted probabilities and original features
        X_meta = self.meta_data_transformer.transform(X, y_pred_prob)
        # Return the final predictions from the meta-model
        return self.meta_model.predict(X_meta)

    def predict_proba(self, X):
        """
        Get predicted probabilities using the base model.
        
        This method exposes the base model probabilities directly.
        
        Parameters:
            X (pandas.DataFrame): The input features.
            
        Returns:
            array-like: The predicted probabilities for the positive class.
        """
        if self.verbose:
            print("[meta_predictions.py] Retrieving predicted probabilities from the base model.")
        return self.base_model.predict_proba(X)[:, 1]
