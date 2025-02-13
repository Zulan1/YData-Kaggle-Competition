import pandas as pd
import numpy as np
import constants as cons
from feature_definitions import Feature
from typing import List


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
    """
    For each user, computes the number of duplicate sessions that share the same timestamp on each day,
    and returns the maximum duplicate count over all days, clipped at an upper bound of 10.

    Process:
      1. Extract the session_date from DateTime.
      2. For each (user, session_date, DateTime) group, compute the duplicate count (i.e. count - 1).
      3. For each (user, session_date), calculate the maximum duplicate count for that day.
      4. For each user, take the maximum duplicate count over all days.
      5. Clip the result to a maximum of 10.
    """
    ddf = df.copy()
    # Extract the session date from DateTime
    ddf["session_date"] = ddf["DateTime"].dt.date

    # Compute the duplicate count for each row by grouping by user, session_date, and DateTime; subtract 1 for the current row.
    ddf["dup_count"] = ddf.groupby(["user_id", "session_date", "DateTime"])["DateTime"].transform("count") - 1

    # For each user and each day, get the maximum duplicate count on that day.
    ddf["max_dup_day"] = ddf.groupby(["user_id", "session_date"])["dup_count"].transform("max")

    # For each user, take the maximum duplicate count among all days.
    ddf["max_dup_over_days"] = ddf.groupby("user_id")["max_dup_day"].transform("max")

    # Clip the maximum duplicate count at 10.
    series = ddf["max_dup_over_days"].clip(upper=10)
    series.name = "duplicate_timestamp_session_count"
    return series.astype(int)

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

# U2. is_high_volume_user
def is_high_volume_user_func(df: pd.DataFrame) -> pd.Series:
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    ddf["daily_count"] = ddf.groupby(["session_date", "user_id"])["session_date"].transform("size")
    ddf["daily_threshold"] = ddf.groupby("session_date")["daily_count"].transform(lambda x: x.quantile(0.95))
    series = (ddf["daily_count"] >= ddf["daily_threshold"]).astype(int)
    series.name = "is_high_volume_user"
    return series

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

# U4c. most_common_time_of_day
def most_common_time_of_day_func(df: pd.DataFrame) -> pd.Series:
    session_time = session_time_of_day_func(df)
    if session_time.isnull().any():
        raise ValueError("NaN detected in session_time_of_day; input must not contain NaN values.")
    user_mode = session_time.groupby(df["user_id"]).agg(lambda x: x.mode().iloc[0])
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
    threshold = max_dup_by_user.quantile(0.99)
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
    """
    Calculates the average daily diversity of products for each user.
    
    For every day a user is active, the number of unique products is computed.
    Then, for each user, the average diversity is taken over all active days,
    rounded up to the nearest integer and clipped at a maximum of 5.
    
    This is a user-based, non-categorical feature.
    """
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    # Calculate daily product diversity for each user.
    daily_diversity = ddf.groupby(["user_id", "session_date"])["product"].nunique()
    # Compute the average daily diversity per user.
    avg_diversity = daily_diversity.groupby("user_id").mean()
    # Round up and clip at 5.
    avg_diversity_ceiled = np.ceil(avg_diversity).clip(upper=5)
    # Map the computed value back to each user.
    result = ddf["user_id"].map(avg_diversity_ceiled)
    result.name = "user_product_diversity"
    return result.astype(int)

# U9b. user_campaign_diversity
def user_campaign_diversity_func(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the average daily diversity of campaigns for each user.
    
    For every day a user is active, the number of unique campaigns is computed.
    Then, for each user, the average diversity is taken over all active days,
    rounded up to the nearest integer and clipped at a maximum of 5.
    
    This is a user-based, non-categorical feature.
    """
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    # Calculate daily campaign diversity for each user.
    daily_diversity = ddf.groupby(["user_id", "session_date"])["campaign_id"].nunique()
    # Compute the average daily diversity per user.
    avg_diversity = daily_diversity.groupby("user_id").mean()
    # Round up and clip at 5.
    avg_diversity_ceiled = np.ceil(avg_diversity).clip(upper=5)
    # Map the computed value back to each user.
    result = ddf["user_id"].map(avg_diversity_ceiled)
    result.name = "user_campaign_diversity"
    return result.astype(int)

# U9c. user_webpage_diversity
def user_webpage_diversity_func(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the average daily diversity of webpages for each user.
    
    For every day a user is active, the number of unique webpages is computed.
    Then, for each user, the average diversity is taken over all active days,
    rounded up to the nearest integer and clipped at a maximum of 5.
    
    This is a user-based, non-categorical feature.
    """
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    # Calculate daily webpage diversity for each user.
    daily_diversity = ddf.groupby(["user_id", "session_date"])["webpage_id"].nunique()
    # Compute the average daily diversity per user.
    avg_diversity = daily_diversity.groupby("user_id").mean()
    # Round up and clip at 5.
    avg_diversity_ceiled = np.ceil(avg_diversity).clip(upper=5)
    # Map the computed value back to each user.
    result = ddf["user_id"].map(avg_diversity_ceiled)
    result.name = "user_webpage_diversity"
    return result.astype(int)

# -----------------------------------------------------------------------------
# New Engineered User-level Feature
# -----------------------------------------------------------------------------
def same_timestamp_session_count_func(df: pd.DataFrame) -> pd.Series:
    """
    For each row, counts the number of sessions (rows) that have the same user_id and the same DateTime.
    The count is clipped at a maximum of 6.
    
    This non-categorical feature is user-based.
    """
    # Count the number of rows per (user_id, DateTime) group.
    count_series = df.groupby(["user_id", "DateTime"])["DateTime"].transform("size")
    # Clip the count at an upper bound of 6.
    count_series = count_series.clip(upper=6)
    count_series.name = "same_timestamp_session_count"
    return count_series.astype(int)

# -----------------------------------------------------------------------------
# New User-based Feature
# -----------------------------------------------------------------------------
def max_daily_user_activity_spread_func(df: pd.DataFrame) -> pd.Series:
    """
    For each user, computes the maximum number of distinct hours during which the user was active in any single day.
    
    Process:
      1. Extract the session date and the hour (0-23) from DateTime.
      2. For each user and day, count the number of unique hours in which the user was active.
      3. For each user, return the maximum count over all days.
    
    This non-categorical user-based feature reflects the daily spread of user activity.
    """
    ddf = df.copy()
    ddf["session_date"] = ddf["DateTime"].dt.date
    ddf["active_hour"] = ddf["DateTime"].dt.hour  # values in 0-23.
    # Count unique active hours per user per day.
    daily_activity_spread = ddf.groupby(["user_id", "session_date"])["active_hour"].nunique()
    # For each user, take the maximum spread over all days.
    max_spread = daily_activity_spread.groupby("user_id").max()
    # Map the computed value back to the original DataFrame based on user_id.
    result = df["user_id"].map(max_spread)
    result.name = "max_daily_user_activity_spread"
    return result.fillna(0).astype(int)

def max_same_timestamp_session_count_func(df: pd.DataFrame) -> pd.Series:
    """
    For each user, computes the maximum number of sessions that share the exact same timestamp.
    
    Process:
      1. Group the DataFrame by ['user_id', 'DateTime'] and count the number of sessions in each group.
      2. For each user, take the maximum count over all such groups.
      3. Clip this maximum at an upper bound of 6.
      
    Returns a non-categorical user-based feature as a Series named "max_same_timestamp_session_count".
    """
    # Group by user_id and DateTime and count the sessions.
    group_counts = df.groupby(["user_id", "DateTime"]).size()
    # For each user, compute the maximum count among all timestamps.
    max_counts = group_counts.groupby("user_id").max()
    # Clip the maximum value at 6.
    max_counts = max_counts.clip(upper=6)
    # Map the computed value back to every row by user_id.
    result = df["user_id"].map(max_counts)
    result.name = "max_same_timestamp_session_count"
    return result.fillna(0).astype(int)

def many_identical_sessions_func(df: pd.DataFrame) -> pd.Series:
    """
    For each user, calculates the maximum number of sessions that are identical in terms of product, campaign_id, and webpage_id.
    
    Process:
      1. Group the DataFrame by ['user_id', 'product', 'campaign_id', 'webpage_id'] and count the sessions in each group.
      2. For each user, take the maximum count among all groups (i.e., the largest N such that the user had N identical sessions).
      3. Clip this maximum at an upper bound of 6.
      
    Returns a non-categorical user-based feature named "many_identical_sessions".
    """
    group_counts = df.groupby(["user_id", "product", "campaign_id", "webpage_id"]).size()
    max_counts = group_counts.groupby("user_id").max()
    max_counts = max_counts.clip(upper=6)
    result = df["user_id"].map(max_counts)
    result.name = "many_identical_sessions"
    return result.fillna(0).astype(int)

def mixed_city_development_index_known_func(df: pd.DataFrame) -> pd.Series:
    """
    For each user, determines if the 'city_development_index_known' field
    has both True and False values across sessions.
    
    Returns 1 if the user has at least one session with city_development_index_known=True 
    and at least one session with city_development_index_known=False.
    Otherwise, returns 0.
    
    This is a binary, non-categorical user-level feature.
    """
    groups = df.groupby("user_id")["city_development_index_known"].apply(lambda s: set(s.dropna()))
    mixed = groups.apply(lambda x: 1 if (True in x and False in x) else 0)
    result = df["user_id"].map(mixed)
    result.name = "mixed_city_development_index_known"
    return result.astype(int)

def mixed_product_category_2_known_func(df: pd.DataFrame) -> pd.Series:
    """
    For each user, determines if the 'product_category_2_known' field
    has both True and False values across sessions.
    
    Returns 1 if the user has at least one session with product_category_2_known=True 
    and at least one session with product_category_2_known=False.
    Otherwise, returns 0.
    
    This is a binary, non-categorical user-level feature.
    """
    groups = df.groupby("user_id")["secondary_product_category_known"].apply(lambda s: set(s.dropna()))
    mixed = groups.apply(lambda x: 1 if (True in x and False in x) else 0)
    result = df["user_id"].map(mixed)
    result.name = "mixed_product_category_2_known"
    return result.astype(int)

# -----------------------------------------------------------------------------
# New Session-based Feature
# -----------------------------------------------------------------------------
def session_mixed_secondary_product_category_known_func(df: pd.DataFrame) -> pd.Series:
    """
    For each session, checks if there exists another session with the same user_id and DateTime
    where the value of 'secondary_product_category_known' is different.
    
    Returns 1 if the group of sessions (sharing the same timestamp) has at least 2 unique values 
    for 'secondary_product_category_known'; otherwise, returns 0.
    
    This is a binary, non-categorical, session-based feature.
    """
    series = df.groupby(["user_id", "DateTime"])["secondary_product_category_known"]\
               .transform(lambda x: int(x.nunique() > 1))
    series.name = "session_mixed_secondary_product_category_known"
    return series.astype(int)

def session_mixed_city_development_index_known_func(df: pd.DataFrame) -> pd.Series:
    """
    For each session, checks if there exists another session with the same user_id and DateTime
    where the value of 'city_development_index_known' is different.
    
    Returns 1 if the group of sessions (sharing the same timestamp) has at least 2 unique values 
    for 'city_development_index_known'; otherwise, returns 0.
    
    This is a binary, non-categorical, session-based feature.
    """
    series = df.groupby(["user_id", "DateTime"])["city_development_index_known"]\
               .transform(lambda x: int(x.nunique() > 1))
    series.name = "session_mixed_city_development_index_known"
    return series.astype(int)

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
    {"name": "user_depth", "scope": "user", "categorical": True, "func": make_original_feature_func("user_depth")},
    {"name": "age_level", "scope": "user", "categorical": False, "func": make_original_feature_func("age_level")},
    {"name": "secondary_product_category_known", "scope": "session", "categorical": False, "func": make_original_feature_func("secondary_product_category_known")},
    {"name": "city_development_index_known", "scope": "session", "categorical": False, "func": make_original_feature_func("city_development_index_known")},

    # --- Engineered Session-level Features ---
    {"name": "daily_sessions_count", "scope": "session", "categorical": False, "func": daily_sessions_count_func},
    {"name": "has_viewed_category_before", "scope": "session", "categorical": False, "func": has_viewed_category_before_func},
    {"name": "has_viewed_campaign_before", "scope": "session", "categorical": False, "func": has_viewed_campaign_before_func},
    {"name": "daily_webpage_count", "scope": "session", "categorical": False, "func": daily_webpage_count_func},
    {"name": "prev_session_same_product", "scope": "session", "categorical": False, "func": prev_session_same_product_func},
    {"name": "session_time_of_day", "scope": "session", "categorical": True, "func": session_time_of_day_func},
    {"name": "duplicate_timestamp_session_count", "scope": "session", "categorical": False, "func": duplicate_timestamp_session_count_func},
    {"name": "product_diversity_same_timestamp", "scope": "session", "categorical": False, "func": product_diversity_same_timestamp_func},
    {"name": "webpage_diversity_same_timestamp", "scope": "session", "categorical": False, "func": webpage_diversity_same_timestamp_func},
    {"name": "session_mixed_secondary_product_category_known", "scope": "session", "categorical": False, "func": session_mixed_secondary_product_category_known_func},
    {"name": "session_mixed_city_development_index_known", "scope": "session", "categorical": False, "func": session_mixed_city_development_index_known_func},

    # --- Engineered User-level Features ---
    {"name": "is_high_volume_user", "scope": "user", "categorical": False, "func": is_high_volume_user_func},
    {"name": "distinct_products_count", "scope": "user", "categorical": False, "func": distinct_products_count_func},
    {"name": "distinct_categories_count", "scope": "user", "categorical": False, "func": distinct_categories_count_func},
    {"name": "distinct_campaigns_count", "scope": "user", "categorical": False, "func": distinct_campaigns_count_func},
    {"name": "distinct_webpages_count", "scope": "user", "categorical": False, "func": distinct_webpages_count_func},
    {"name": "most_common_time_of_day", "scope": "user", "categorical": True, "func": most_common_time_of_day_func},
    {"name": "total_active_days", "scope": "user", "categorical": False, "func": total_active_days_func},
    {"name": "sessions_with_min_gap_count", "scope": "user", "categorical": False, "func": sessions_with_min_gap_count_func},
    {"name": "excess_duplicate_timestamps", "scope": "user", "categorical": False, "func": excess_duplicate_timestamps_func},
    {"name": "early_morning_sessions_count", "scope": "user", "categorical": False, "func": early_morning_sessions_count_func},
    {"name": "user_product_diversity", "scope": "user", "categorical": False, "func": user_product_diversity_func},
    {"name": "user_campaign_diversity", "scope": "user", "categorical": False, "func": user_campaign_diversity_func},
    {"name": "user_webpage_diversity", "scope": "user", "categorical": False, "func": user_webpage_diversity_func},
    {"name": "same_timestamp_session_count", "scope": "user", "categorical": False, "func": same_timestamp_session_count_func},
    {"name": "max_daily_user_activity_spread", "scope": "user", "categorical": False, "func": max_daily_user_activity_spread_func},
    {"name": "max_same_timestamp_session_count", "scope": "user", "categorical": False, "func": max_same_timestamp_session_count_func},
    {"name": "many_identical_sessions", "scope": "user", "categorical": False, "func": many_identical_sessions_func},
    {"name": "mixed_city_development_index_known", "scope": "user", "categorical": False, "func": mixed_city_development_index_known_func},
    {"name": "mixed_product_category_2_known", "scope": "user", "categorical": False, "func": mixed_product_category_2_known_func},
]

def cast_features(df_final: pd.DataFrame, features: List[Feature]) -> pd.DataFrame:
    """
    Casts the columns in df_final based on the dtype specified in each Feature instance.
    For categorical features, it converts to Pandas' string dtype; for others, to int.
    """
    for feature in features:
        col = feature.name
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(feature.dtype)
    return df_final

def prepare_features(df: pd.DataFrame, verbose: bool = False) -> tuple:
    """
    Given a DataFrame with all necessary columns, compute and return a new DataFrame 
    that contains exactly the final features (ordered as in FEATURES_LIST) with their correct dtypes,
    while preserving the original DataFrame's index.
    
    Also returns a list of Feature instances describing each feature added.
    (This list helps the data transformer know which features are categorical for downstream tasks.)
    
    Processing:
      - Renames columns if necessary.
      - Parses DateTime.
      - Computes each feature using its corresponding function.
      - Maps the computed features to a new DataFrame with the correct index.
      - Creates Feature instances from the definition list.
      - Casts each feature column using the dtype declared by its Feature.
      - Adds the target column (cast to int) if present.
    """
    TARGET_COLUMN = cons.TARGET_COLUMN

    # Rename product_category_1 if necessary.
    if "product_category_1" in df.columns and "product_category" not in df.columns:
        df = df.rename(columns={"product_category_1": "product_category"})

    # Ensure DateTime is parsed.
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    # Create an empty DataFrame with the same index.
    df_final = pd.DataFrame(index=df.index)

    # Process each feature as specified in FEATURES_LIST.
    for feat in FEATURES_LIST:
        feature_name = feat["name"]
        series = feat["func"](df)
        # Reindex to ensure alignment.
        series = series.reindex(df.index)
        df_final[feature_name] = series

    # Add the target column if available.
    if TARGET_COLUMN in df.columns:
        df_final[TARGET_COLUMN] = df[TARGET_COLUMN].reindex(df.index)

    # Create Feature instances.
    features_added = [
        Feature(name=feat["name"], scope=feat["scope"], categorical=feat["categorical"])
        for feat in FEATURES_LIST
    ]

    # Modular casting: uniformly cast features based on each Feature's dtype.
    df_final = cast_features(df_final, features_added)

    # Ensure the target column is cast to int.
    if TARGET_COLUMN in df_final.columns:
        df_final[TARGET_COLUMN] = df_final[TARGET_COLUMN].astype(int)

    if verbose:
        print("[feature_engineering.py] Feature engineering completed")
        print(f"[feature_engineering.py] Final DataFrame has {len(df_final.columns)} columns with original index preserved.")

    return df_final, features_added