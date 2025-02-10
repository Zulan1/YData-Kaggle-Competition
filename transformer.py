import pandas as pd
import constants as cons
from impute import ClickImputer
from sklearn.preprocessing import OneHotEncoder
from feature_engineering import add_features
import config as conf
from catboost_transform import catboost_transform


class DataTransformer:
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.click_imputer = ClickImputer()
    def one_hot_encode(self, df: pd.DataFrame):
        encoded_cats = self.ohe.transform(df[cons.COLUMNS_TO_OHE])
        feature_names = self.ohe.get_feature_names_out(cons.COLUMNS_TO_OHE)
        encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
        df = df.drop(columns=cons.COLUMNS_TO_OHE)
        df = pd.concat([df, encoded_df], axis=1)
        return df
    def one_hot_decode(self, df: pd.DataFrame):
        """Decode one-hot encoded columns back to their original categorical values."""
        inverse_transformed = self.ohe.inverse_transform(df[self.ohe.get_feature_names_out(cons.COLUMNS_TO_OHE)])
        decoded_df = pd.DataFrame(inverse_transformed, columns=cons.COLUMNS_TO_OHE, index=df.index)
        df = df.drop(columns=self.ohe.get_feature_names_out(cons.COLUMNS_TO_OHE))
        df = pd.concat([df, decoded_df], axis=1)
        return df
    def fit(self, df: pd.DataFrame):
        self.click_imputer.fit(df)
        self.ohe.fit(df[cons.COLUMNS_TO_OHE])
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self.click_imputer.transform(df)
        df = add_features(df, add_catboost_features=conf.USE_CATBOOST)
        if conf.USE_CATBOOST:
            df = catboost_transform(df)
        else:
            df = self.one_hot_encode(df)
        df = df.drop(columns=cons.INDEX_COLUMNS + [cons.DATETIME_COLUMN])
        return df
    
    
    
    
        