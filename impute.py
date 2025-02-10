from sklearn.impute import KNNImputer, SimpleImputer
import constants as cons
import config as conf
import pandas as pd
import numpy as np

def remove_outliers(df):
    """Remove outliers from the dataframe by dropping users with high session counts."""
    # Count sessions per user
    session_counts = df['user_id'].value_counts()
    
    # Find cutoff point at 99th percentile
    cutoff = np.percentile(session_counts, 95)
    
    # Get users below the 99th percentile
    valid_users = session_counts[session_counts <= cutoff].index
    
    # Filter dataframe to only include valid users
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    return df_filtered

class ClickImputer:
    """Handles imputation of missing values in click data."""
    
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent')
    
    def fit(self, df: pd.DataFrame):
        """Fit the imputer on training data."""
        self.columns_to_impute = list(set(df.columns) - set(cons.INDEX_COLUMNS + [cons.TARGET_COLUMN]))
        self.imputer.fit(df[self.columns_to_impute])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        df = df.copy()
        df = self._impute_user_id(df)
        df = self._impute_session_id(df)
        df[self.columns_to_impute] = self.imputer.transform(df[self.columns_to_impute])
        return df
    
    @staticmethod
    def _impute_user_id(df: pd.DataFrame) -> pd.DataFrame:
        """Impute user_id column with new unique values."""
        if df['user_id'].isna().any():
            max_id = df['user_id'].max() if not df['user_id'].isna().all() else 0
            missing_mask = df['user_id'].isna()
            df.loc[missing_mask, 'user_id'] = range(
                int(max_id) + 1,
                int(max_id) + 1 + missing_mask.sum()
            )
        return df
    
    @staticmethod
    def _impute_session_id(df: pd.DataFrame) -> pd.DataFrame:
        """Impute session_id column with new unique values."""
        if df['session_id'].isna().any():
            max_id = df['session_id'].max() if not df['session_id'].isna().all() else 0
            missing_mask = df['session_id'].isna()
            df.loc[missing_mask, 'session_id'] = range(
                int(max_id) + 1,
                int(max_id) + 1 + missing_mask.sum()
            )
        return df
