import pandas as pd
import constants as cons
from impute import ClickImputer
from sklearn.preprocessing import OneHotEncoder
from feature_engineering import add_features
import config as conf
from catboost_transform import catboost_transform
from feature_registry import FeatureRegistry

class DataTransformer:
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.click_imputer = ClickImputer()
        self.feature_registry = FeatureRegistry()
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
        df = self.click_imputer.transform(df)
        df, self.feature_registry = add_features(df, add_catboost_features=conf.USE_CATBOOST)
        if conf.USE_CATBOOST:
            df = catboost_transform(df)
        else:
            df = self.one_hot_encode(df)
        df = df.drop(columns=cons.INDEX_COLUMNS + [cons.DATETIME_COLUMN])
        return df
    
    
    
    
        