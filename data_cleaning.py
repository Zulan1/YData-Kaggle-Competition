import pandas as pd
import constants as cons
import numpy as np

"""
Data cleaning class. Drop missing values and duplicates. 
For columns with high missing values, keep track of whether the value is known or not.
"""

class DataCleaner:
    def __init__(self, verbose: bool = False, mode: str = 'train'):
        self.verbose = verbose
        self.mode = mode  # Store the mode so it can be used later.
        self.columns_to_drop = cons.COLUMNS_TO_DROP

    def filter_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df['age_level'] == 0, 'age_level'] = np.nan
        initial_rows = len(df)
        df = df.dropna(subset=['session_id'])
        if self.verbose:
            print(f"[DataCleaner] Dropped {initial_rows - len(df)} rows containing NaN values in 'session_id' column.")
        df = df.drop_duplicates()
        initial_rows = len(df)
        duplicates_removed = initial_rows - len(df)
        if self.verbose:
            print(f"[DataCleaner] Dropped {duplicates_removed} duplicate rows.")
        rows_before_na = len(df)
        df = df.dropna()
        na_removed = rows_before_na - len(df)
        if self.verbose:
            print(f"[DataCleaner] Dropped {na_removed} rows containing NaN values.")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print("[DataCleaner] Starting data cleaning process...")

        # Drop unwanted columns.
        df['secondary_product_category_known'] = df['product_category_2'].notna()
        df['city_development_index_known'] = df['city_development_index'].notna()
        df = df.drop(columns=self.columns_to_drop)
        if self.verbose:
            print(f"[DataCleaner] Dropped columns: {self.columns_to_drop}")

        # If the cleaning is for training, apply additional filtering.
        if self.mode == 'train':
            if self.verbose:
                print("[DataCleaner] Mode is 'train': Filtering training data...")
            df = self.filter_training_data(df)

        if self.verbose:
            print("[DataCleaner] Data cleaning process completed.")

        return df
