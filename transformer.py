import pandas as pd
import constants as cons
from impute import ClickImputer
from sklearn.preprocessing import OneHotEncoder
import config as conf
from feature_engineering import prepare_features

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(int)
    return df

class DataTransformer:
    def __init__(self, verbose=False):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.click_imputer = ClickImputer()
        self.verbose = verbose
        self.registry = None

    def one_hot_encode(self, df: pd.DataFrame):
        encoded_cats = self.ohe.transform(df[cons.COLUMNS_TO_OHE])
        feature_names = self.ohe.get_feature_names_out(cons.COLUMNS_TO_OHE)
        encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
        df = df.drop(columns=cons.COLUMNS_TO_OHE)
        df = pd.concat([df, encoded_df], axis=1)
        return df
    def fit(self, df: pd.DataFrame):
        self.click_imputer.fit(df)
        self.ohe.fit(df[cons.COLUMNS_TO_OHE])
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = preprocess_data(df)
        df = self.click_imputer.transform(df)
        df = prepare_features(df, verbose=self.verbose)
        if not conf.USE_CATBOOST:
            df = self.one_hot_encode(df)
        return df
    
    
    
    
        