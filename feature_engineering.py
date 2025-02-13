import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Helper to generate function for original features.
# -----------------------------------------------------------------------------
def make_original_feature_func(col: str):
    return lambda df: df[col]

# -----------------------------------------------------------------------------
# Engineered feature functions
# Each function must be named as <feature_name>_func and produce a Series named <feature_name>
# -----------------------------------------------------------------------------

# S1. daily_sessions_count
def daily_sessions_count_func(df: pd.DataFrame) -> pd.Series:
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    series = ddf.groupby(["user_id", "session_date"])["session_date"].transform("count").clip(upper=10)
    series.name = "daily_sessions_count"
    return series

# S2. has_viewed_product_before
def has_viewed_product_before_func(df: pd.DataFrame) -> pd.Series:
    df_sorted = df.sort_values(["user_id", "DateTime"])
    series = df_sorted.groupby(["user_id", "product"]).cumcount() > 0
    series.name = "has_viewed_product_before"
    return series.astype(int)

# S3. has_viewed_category_before
def has_viewed_category_before_func(df: pd.DataFrame) -> pd.Series:
    df_sorted = df.sort_values(["user_id", "DateTime"])
    series = df_sorted.groupby(["user_id", "product_category"]).cumcount() > 0
    series.name = "has_viewed_category_before"
    return series.astype(int)

# S4. has_viewed_campaign_before
def has_viewed_campaign_before_func(df: pd.DataFrame) -> pd.Series:
    df_sorted = df.sort_values(["user_id", "DateTime"])
    series = df_sorted.groupby(["user_id", "campaign_id"]).cumcount() > 0
    series.name = "has_viewed_campaign_before"
    return series.astype(int)

# S5. daily_webpage_count
def daily_webpage_count_func(df: pd.DataFrame) -> pd.Series:
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    series = ddf.groupby(["user_id", "session_date"])["webpage_id"].transform("count").clip(upper=10)
    series.name = "daily_webpage_count"
    return series

# S6. prev_session_same_product
def prev_session_same_product_func(df: pd.DataFrame) -> pd.Series:
    df_sorted = df.sort_values(["user_id", "DateTime"])
    series = df_sorted.groupby("user_id")["product"].shift() == df_sorted["product"]
    series.name = "prev_session_same_product"
    return series.fillna(False).astype(int)

# S7. session_time_of_day (5 bins: 8:00-13:00, 13:00-18:00, 18:00-21:00, 21:00-1:00, 1:00-8:00)
def session_time_of_day_func(df: pd.DataFrame) -> pd.Series:
    hour = df["DateTime"].dt.hour
    conditions = [
        (hour >= 8) & (hour < 13),
        (hour >= 13) & (hour < 18),
        (hour >= 18) & (hour < 21),
        ((hour >= 21) | (hour < 1)),
        (hour >= 1) & (hour < 8)
    ]
    choices = ["Morning", "Afternoon", "Evening", "Night", "Early Morning"]
    series = pd.Series(np.select(conditions, choices, default="Unknown"), index=df.index)
    series.name = "session_time_of_day"
    return series.astype(str)

# S8. session_day_of_week
def session_day_of_week_func(df: pd.DataFrame) -> pd.Series:
    series = df["DateTime"].dt.dayofweek.astype(str)
    series.name = "session_day_of_week"
    return series

# S9. next_day_session_indicator
def next_day_session_indicator_func(df: pd.DataFrame) -> pd.Series:
    ddf = df.sort_values(["user_id", "DateTime"])
    session_date = ddf["DateTime"].dt.date
    shifted = ddf.groupby("user_id")["DateTime"].shift(-1).dt.date
    series = (shifted > session_date).astype(int)
    series.name = "next_day_session_indicator"
    return series.fillna(0)

# S10. daily_session_sequence
def daily_session_sequence_func(df: pd.DataFrame) -> pd.Series:
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    series = ddf.groupby(["user_id", "session_date"]).cumcount() + 1
    series = series.clip(upper=7)
    series.name = "daily_session_sequence"
    return series

# S11. duplicate_timestamp_session_count
def duplicate_timestamp_session_count_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby(["user_id", "DateTime"])["DateTime"].transform("count") - 1
    series.name = "duplicate_timestamp_session_count"
    return series.clip(lower=0)

# S12. product_diversity_same_timestamp
def product_diversity_same_timestamp_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby(["user_id", "DateTime"])["product"].transform("nunique").clip(upper=5)
    series.name = "product_diversity_same_timestamp"
    return series

# S13. webpage_diversity_same_timestamp
def webpage_diversity_same_timestamp_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby(["user_id", "DateTime"])["webpage_id"].transform("nunique").clip(upper=5)
    series.name = "webpage_diversity_same_timestamp"
    return series

# U1. total_sessions_count
def total_sessions_count_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id").cumcount() + 1
    series = series.clip(upper=10)
    series.name = "total_sessions_count"
    return series

# U2. is_high_volume_user
def is_high_volume_user_func(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("user_id").size()
    threshold = counts.quantile(0.95)
    series = df["user_id"].map(lambda u: 1 if counts.loc[u] >= threshold else 0)
    series.name = "is_high_volume_user"
    return series.astype(int)

# U3a. distinct_products_count
def distinct_products_count_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["product"].transform("nunique").clip(upper=4)
    series.name = "distinct_products_count"
    return series

# U3b. distinct_categories_count
def distinct_categories_count_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["product_category"].transform("nunique").clip(upper=4)
    series.name = "distinct_categories_count"
    return series

# U3c. distinct_campaigns_count
def distinct_campaigns_count_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["campaign_id"].transform("nunique").clip(upper=4)
    series.name = "distinct_campaigns_count"
    return series

# U3d. distinct_webpages_count
def distinct_webpages_count_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["webpage_id"].transform("nunique").clip(upper=4)
    series.name = "distinct_webpages_count"
    return series

# U4a. most_common_category
def most_common_category_func(df: pd.DataFrame) -> pd.Series:
    """
    For each row, returns the mode of the product_category values across all rows for that user.
    
    Assumes that the product_category column contains no NaNs.
    Raises ValueError if any NaNs are detected.
    """
    # Check for NaNs in product_category.
    if df["product_category"].isnull().any():
        raise ValueError("NaN detected in product_category; input must not contain NaN values.")
    
    # Group by user_id and compute the mode.
    user_mode = df.groupby("user_id")["product_category"].agg(lambda x: x.mode().iloc[0])
    
    # Map the computed mode back to each row based on user_id.
    result = df["user_id"].map(user_mode)
    result.name = "most_common_category"
    return result.astype("string")

# U4b. most_common_campaign
def most_common_campaign_func(df: pd.DataFrame) -> pd.Series:
    """
    For each row, returns the mode of the campaign_id values across all rows for that user.
    
    Assumes that the campaign_id column contains no NaNs.
    Raises ValueError if any NaNs are detected.
    """
    # Check for NaNs in campaign_id.
    if df["campaign_id"].isnull().any():
        raise ValueError("NaN detected in campaign_id; input must not contain NaN values.")
    
    # Group by user_id and compute the mode.
    user_mode = df.groupby("user_id")["campaign_id"].agg(lambda x: x.mode().iloc[0])
    
    # Map the computed mode back to each row based on user_id.
    result = df["user_id"].map(user_mode)
    result.name = "most_common_campaign"
    return result.astype("string")

# U4c. most_common_time_of_day
def most_common_time_of_day_func(df: pd.DataFrame) -> pd.Series:
    """
    For each row, returns the mode (or one of the modes) of the session_time_of_day values
    across all rows for that user.
    
    Assumes that session_time_of_day (computed from DateTime) contains no NaNs.
    Raises ValueError if any NaNs are detected.
    """
    # Compute the session_time_of_day using the previously defined function.
    session_time = session_time_of_day_func(df)
    
    # Ensure no NaNs exist.
    if session_time.isnull().any():
        raise ValueError("NaN detected in session_time_of_day; input must not contain NaN values.")
    
    # Group by user_id and compute the mode.
    user_mode = session_time.groupby(df["user_id"]).agg(lambda x: x.mode().iloc[0])
    
    # Map the computed user-level mode back onto each row.
    result = df["user_id"].map(user_mode)
    result.name = "most_common_time_of_day"
    return result.astype("string")

# U5. total_active_days
def total_active_days_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["DateTime"].transform(lambda x: x.dt.date.nunique())
    series.name = "total_active_days"
    return series

# U6. sessions_with_min_gap_count
def sessions_with_min_gap_count_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["DateTime"].transform(lambda x: (x.diff().dt.total_seconds() < 300).sum())
    series = series.clip(upper=5)
    series.name = "sessions_with_min_gap_count"
    return series

# U7. excess_duplicate_timestamps
def excess_duplicate_timestamps_func(df: pd.DataFrame) -> pd.Series:
    dup_counts = df.groupby(["user_id", "DateTime"]).size().reset_index(name="count")
    max_dup_by_user = dup_counts.groupby("user_id")["count"].max()
    threshold = max_dup_by_user.quantile(0.95)
    series = df["user_id"].map(lambda u: 1 if max_dup_by_user.loc[u] >= threshold else 0)
    series.name = "excess_duplicate_timestamps"
    return series.astype(int)

# U8. early_morning_sessions_count
def early_morning_sessions_count_func(df: pd.DataFrame) -> pd.Series:
    condition = ((df["DateTime"].dt.hour > 1) & (df["DateTime"].dt.hour < 6)) | \
                ((df["DateTime"].dt.hour == 6) & (df["DateTime"].dt.minute <= 30))
    series = df.groupby("user_id")["DateTime"].transform(lambda x: condition.loc[x.index].sum())
    series = series.clip(upper=4)
    series.name = "early_morning_sessions_count"
    return series

# U9a. user_product_diversity
def user_product_diversity_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["product"].transform("nunique").clip(upper=5)
    series.name = "user_product_diversity"
    return series

# U9b. user_campaign_diversity
def user_campaign_diversity_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["campaign_id"].transform("nunique").clip(upper=5)
    series.name = "user_campaign_diversity"
    return series

# U9c. user_webpage_diversity
def user_webpage_diversity_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby("user_id")["webpage_id"].transform("nunique").clip(upper=5)
    series.name = "user_webpage_diversity"
    return series

# -----------------------------------------------------------------------------
# Definition of the final feature list including function pointers.
# -----------------------------------------------------------------------------
FEATURES_LIST = [
    # --- Original Features ---
    {"name": "product", "scope": "session", "categorical": True, "func": make_original_feature_func("product")},
    {"name": "product_category", "scope": "session", "categorical": True, "func": make_original_feature_func("product_category")},
    {"name": "campaign_id", "scope": "session", "categorical": True, "func": make_original_feature_func("campaign_id")},
    {"name": "webpage_id", "scope": "session", "categorical": True, "func": make_original_feature_func("webpage_id")},
    {"name": "user_group_id", "scope": "user", "categorical": False, "func": make_original_feature_func("user_group_id")},
    {"name": "gender", "scope": "user", "categorical": True, "func": make_original_feature_func("gender")},
    {"name": "var_1", "scope": "session", "categorical": False, "func": make_original_feature_func("var_1")},
    {"name": "user_depth", "scope": "user", "categorical": False, "func": make_original_feature_func("user_depth")},
    {"name": "age_level", "scope": "user", "categorical": False, "func": make_original_feature_func("age_level")},

    # --- Engineered Session-level Features (S1–S13) ---
    {"name": "daily_sessions_count", "scope": "session", "categorical": False, "func": daily_sessions_count_func},           # S1
    {"name": "has_viewed_product_before", "scope": "session", "categorical": False, "func": has_viewed_product_before_func},    # S2
    {"name": "has_viewed_category_before", "scope": "session", "categorical": False, "func": has_viewed_category_before_func},  # S3
    {"name": "has_viewed_campaign_before", "scope": "session", "categorical": False, "func": has_viewed_campaign_before_func},  # S4
    {"name": "daily_webpage_count", "scope": "session", "categorical": False, "func": daily_webpage_count_func},              # S5
    {"name": "prev_session_same_product", "scope": "session", "categorical": False, "func": prev_session_same_product_func},    # S6
    {"name": "session_time_of_day", "scope": "session", "categorical": True,  "func": session_time_of_day_func},               # S7
    {"name": "session_day_of_week", "scope": "session", "categorical": True,  "func": session_day_of_week_func},               # S8
    {"name": "next_day_session_indicator", "scope": "session", "categorical": False, "func": next_day_session_indicator_func},# S9
    {"name": "daily_session_sequence", "scope": "session", "categorical": False, "func": daily_session_sequence_func},       # S10
    {"name": "duplicate_timestamp_session_count", "scope": "session", "categorical": False, "func": duplicate_timestamp_session_count_func},  # S11
    {"name": "product_diversity_same_timestamp", "scope": "session", "categorical": False, "func": product_diversity_same_timestamp_func}, # S12
    {"name": "webpage_diversity_same_timestamp", "scope": "session", "categorical": False, "func": webpage_diversity_same_timestamp_func}, # S13

    # --- Engineered User-level Features (U1–U9c) ---
    {"name": "total_sessions_count", "scope": "user", "categorical": False, "func": total_sessions_count_func},             # U1
    {"name": "is_high_volume_user", "scope": "user", "categorical": False, "func": is_high_volume_user_func},                # U2
    {"name": "distinct_products_count", "scope": "user", "categorical": False, "func": distinct_products_count_func},         # U3a
    {"name": "distinct_categories_count", "scope": "user", "categorical": False, "func": distinct_categories_count_func},     # U3b
    {"name": "distinct_campaigns_count", "scope": "user", "categorical": False, "func": distinct_campaigns_count_func},       # U3c
    {"name": "distinct_webpages_count", "scope": "user", "categorical": False, "func": distinct_webpages_count_func},         # U3d
    {"name": "most_common_category", "scope": "user", "categorical": True,  "func": most_common_category_func},               # U4a
    {"name": "most_common_campaign", "scope": "user", "categorical": True,  "func": most_common_campaign_func},               # U4b
    {"name": "most_common_time_of_day", "scope": "user", "categorical": True,  "func": most_common_time_of_day_func},            # U4c
    {"name": "total_active_days", "scope": "user", "categorical": False, "func": total_active_days_func},                     # U5
    {"name": "sessions_with_min_gap_count", "scope": "user", "categorical": False, "func": sessions_with_min_gap_count_func},   # U6
    {"name": "excess_duplicate_timestamps", "scope": "user", "categorical": False, "func": excess_duplicate_timestamps_func},  # U7
    {"name": "early_morning_sessions_count", "scope": "user", "categorical": False, "func": early_morning_sessions_count_func},        # U8
    {"name": "user_product_diversity", "scope": "user", "categorical": False, "func": user_product_diversity_func},             # U9a
    {"name": "user_campaign_diversity", "scope": "user", "categorical": False, "func": user_campaign_diversity_func},           # U9b
    {"name": "user_webpage_diversity", "scope": "user", "categorical": False, "func": user_webpage_diversity_func},             # U9c
]

# -----------------------------------------------------------------------------
# Final function: prepare_features()
# -----------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Given a DataFrame with all necessary columns, compute and return a new DataFrame 
    that contains exactly the final features (ordered as in FEATURES_LIST) while preserving 
    the original DataFrame's index.

    - For each feature in FEATURES_LIST, the corresponding function (under the "func" key) 
      is used to compute the feature.
    - Categorical features are cast to the Pandas string dtype ("string") and non-categorical 
      features to int.
    - If the target column ("is_click") is present, it is added (cast to int).
    """
    TARGET_COLUMN = "is_click"
    
    # Rename product_category_1 if necessary.
    if "product_category_1" in df.columns and "product_category" not in df.columns:
        df = df.rename(columns={"product_category_1": "product_category"})
    
    # Ensure DateTime is parsed as a datetime.
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    
    # Create an empty DataFrame with the same index as the original.
    df_final = pd.DataFrame(index=df.index)
    
    # Compute each feature using its function pointer from FEATURES_LIST.
    for feat in FEATURES_LIST:
        feature_name = feat["name"]
        series = feat["func"](df)
        # Reindex to ensure alignment with original DataFrame
        series = series.reindex(df.index)
        df_final[feature_name] = series

    # Add the target column if available.
    if TARGET_COLUMN in df.columns:
        df_final[TARGET_COLUMN] = df[TARGET_COLUMN].reindex(df.index)
    
    # Enforce data types: categorical features to Pandas string dtype and non-categorical features to int.
    for feat in FEATURES_LIST:
        col = feat["name"]
        if col in df_final.columns:
            if feat["categorical"]:
                df_final[col] = df_final[col].astype("string[python]")
            else:
                df_final[col] = df_final[col].astype(int)
    
    if TARGET_COLUMN in df_final.columns:
        df_final[TARGET_COLUMN] = df_final[TARGET_COLUMN].astype(int)
    
    if verbose:
        print(f"Feature engineering completed")
        print(f"Returning final DataFrame with {len(df_final.columns)} columns and original index preserved.")
    return df_final