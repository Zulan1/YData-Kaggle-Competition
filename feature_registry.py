###############################################################################
#                       Feature & FeatureRegistry classes
###############################################################################

import pandas as pd

class Feature:
    def __init__(self, name, func, scope='session', categorical=False, add_to_df=True):
        """
        Parameters:
          name: str, the name of the feature (or the output column).
          func: callable, a function that takes a DataFrame and returns a Series 
                or DataFrame of the new feature.
          scope: either 'session' or 'user' to indicate which type of feature it is.
          categorical: bool, True if the output is categorical.
          add_to_df: bool, if False the feature is registered but not added to the DataFrame
                     in the transform (useful for original/pass-through features).
        """
        self.name = name
        self.func = func
        self.scope = scope
        self.categorical = categorical
        self.add_to_df = add_to_df

    def apply(self, df):
        # Call the feature function and return a Series or DataFrame.
        return self.func(df)


class FeatureRegistry:
    def __init__(self):
        self.features = []

    def register(self, feature):
        self.features.append(feature)

    def transform(self, df):
        """
        Applies all registered feature functions to df and returns a new DataFrame 
        with the new features appended. If a feature's 'add_to_df' flag is False,
        that feature function is skipped.
        """
        df_transformed = df.copy()
        for feat in self.features:
            if not feat.add_to_df:
                # Skip adding this feature; it already exists.
                continue

            new_feature = feat.apply(df_transformed)
            if isinstance(new_feature, pd.Series):
                # Overwrite any existing column.
                df_transformed[feat.name] = new_feature
            elif isinstance(new_feature, pd.DataFrame):
                # If a DataFrame is produced, add each column.
                for col in new_feature.columns:
                    df_transformed[col] = new_feature[col]
            else:
                raise ValueError("Feature function must return a pandas Series or DataFrame")
        return df_transformed

    def get_feature_names(self, scope=None, categorical=None):
        """
        Returns a list of feature names filtered by scope and/or categorical flag.
        If scope is None, returns all features; similarly for categorical.
        """
        names = []
        for feat in self.features:
            if scope is not None and feat.scope != scope:
                continue
            if categorical is not None and feat.categorical != categorical:
                continue
            names.append(feat.name)
        return names