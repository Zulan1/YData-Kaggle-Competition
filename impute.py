from sklearn.impute import SimpleImputer
import constants as cons
import pandas as pd
import numpy as np

class ClickDataImputer:
    """Handles imputation of missing values in click data."""
    
    def __init__(self, verbose=False):
        """
        Initialize ClickDataImputer.

        Parameters:
            verbose (bool): If True, print additional status messages during transformation.
        """
        self.verbose = verbose
        self.imputer = SimpleImputer(strategy='most_frequent')
    
    def fit(self, df: pd.DataFrame):
        """Fit the imputer on training data."""
        # Determine columns to impute: all columns except index and target.
        self.columns_to_impute = list(set(df.columns) - set(cons.INDEX_COLUMNS + [cons.TARGET_COLUMN]))
        self.imputer.fit(df[self.columns_to_impute])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values.
        
        If self.verbose is True, prints the number of missing values replaced.
        """
        df = df.copy()
        df.loc[df['age_level'] == 0, 'age_level'] = np.nan
        
        # Determine all columns to monitor for missing values.
        impute_columns = set(["user_id", "session_id"]).union(set(self.columns_to_impute))
        missing_before = df[list(impute_columns)].isna().sum().sum()
        
        # Impute user_id and session_id explicitly.
        df = self._impute_user_id(df)
        df = self._impute_session_id(df)
        
        # Impute the remaining columns using SimpleImputer.
        df[self.columns_to_impute] = self.imputer.transform(df[self.columns_to_impute])
        
        missing_after = df[list(impute_columns)].isna().sum().sum()
        imputed_count = missing_before - missing_after
        
        if self.verbose:
            print(f"[ClickDataImputer] Imputed {imputed_count} missing values.")
        
        return df
    
    @staticmethod
    def _impute_user_id(df: pd.DataFrame) -> pd.DataFrame:
        """Impute user_id column with new unique values if missing."""
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
        """Impute session_id column with new unique values if missing."""
        if df['session_id'].isna().any():
            max_id = df['session_id'].max() if not df['session_id'].isna().all() else 0
            missing_mask = df['session_id'].isna()
            df.loc[missing_mask, 'session_id'] = range(
                int(max_id) + 1,
                int(max_id) + 1 + missing_mask.sum()
            )
        return df
