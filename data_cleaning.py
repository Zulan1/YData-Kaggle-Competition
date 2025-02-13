import pandas as pd
import constants as cons

class DataCleaner:
    def __init__(self, verbose: bool = False, mode: str = 'train'):
        self.verbose = verbose
        self.mode = mode  # Store the mode so it can be used later.
        self.columns_to_drop = cons.COLUMNS_TO_DROP

    def filter_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop duplicate rows first.
        initial_rows = len(df)
        df = df.dropna(subset=['session_id'])
        df = df.drop_duplicates()
        df = df.dropna()
        duplicates_removed = initial_rows - len(df)
        if self.verbose:
            print(f"[DataCleaner] Dropped {duplicates_removed} duplicate rows.")

        # Then drop rows containing NaN values.
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
