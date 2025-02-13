import pandas as pd
import constants as cons
from impute import ClickDataImputer
from sklearn.preprocessing import OneHotEncoder
from feature_engineering import prepare_features

def cast_float_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all float columns in the DataFrame to integers.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with float columns cast to integer dtype.
    """
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(int)
    return df

class DataTransformer:
    """
    Handles preprocessing steps including imputation, one-hot encoding,
    and feature engineering.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.ctr_imputer = ClickDataImputer()
        self.features = []

    def one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns defined in cons.COLUMNS_TO_OHE.

        Args:
            df (pd.DataFrame): DataFrame containing the columns to be encoded.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns replacing the originals.
        """
        encoded_array = self.ohe.transform(df[cons.COLUMNS_TO_OHE])
        feature_names = self.ohe.get_feature_names_out(cons.COLUMNS_TO_OHE)
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
        df = df.drop(columns=cons.COLUMNS_TO_OHE)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the imputer and one-hot encoder on the provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit on.
        """
        self.ctr_imputer.fit(df)
        self.ohe.fit(df[cons.COLUMNS_TO_OHE])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by applying imputation, casting float columns to int,
        and performing feature engineering.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        df = df.copy()
        df = self.ctr_imputer.transform(df)
        df = cast_float_columns_to_int(df)
        df, self.features = prepare_features(df, verbose=self.verbose)
        return df
    
    
    
    
        