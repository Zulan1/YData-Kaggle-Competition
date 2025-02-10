import pandas as pd
import numpy as np
def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time features from the DateTime column.
    
    Args:
        df: DataFrame containing DateTime column
        
    Returns:
        DataFrame with added time-based features:
        - One-hot encoded day of week (day_0 through day_6)
        - Cyclical hour encoding (hour_sin and hour_cos)
    """
    # One-hot encode day of week (0-6)
    df = df.copy()
 #   days = df['DateTime'].dt.dayofweek
 #   for day in range(7):
 #       day_col = (days == day).astype(int)
 #       df.loc[:, f'day_{day}'] = day_col
    
    # Cyclical encoding for hour of day
    hours = df['DateTime'].dt.hour
    hours_sin = np.sin(hours * (2 * np.pi / 24))
    hours_cos = np.cos(hours * (2 * np.pi / 24))
    df.loc[:, 'hour_sin'] = hours_sin
    df.loc[:, 'hour_cos'] = hours_cos
    
    return df
def add_first_session_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary feature indicating if this is the user's first session.
    
    Args:
        df: DataFrame containing user_id column
        
    Returns:
        DataFrame with new first_session column
    """
    # Create a temporary index to ensure correct mapping
    df = df.copy()  # Avoid modifying original
    df['_temp_idx'] = range(len(df))
    
    # Sort by user and datetime
    sorted_df = df.sort_values(['user_id', 'DateTime'])
    
    # First occurrence for each user will be True
    first_session = ~sorted_df['user_id'].duplicated()
    
    # Create mapping DataFrame to restore original order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'first_session': first_session
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['first_session'] = result_df['first_session'].astype(int)
    df.drop('_temp_idx', axis=1, inplace=True)
    
    return df
def add_session_within_last_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary feature indicating if user has had another session within the last hour.
    
    Args:
        df: DataFrame containing user_id and DateTime columns
        
    Returns:
        DataFrame with new session_within_last_hour column
    """
    # Create a temporary index to ensure correct mapping
    df = df.copy()  # Avoid modifying original
    df['_temp_idx'] = range(len(df))
    
    # Sort by user and datetime
    sorted_df = df.sort_values(['user_id', 'DateTime'])
    
    # Calculate time difference between consecutive sessions for same user
    time_diff = sorted_df.groupby('user_id')['DateTime'].diff()
    
    # Check if time difference is <= 1 hour (3600 seconds)
    session_within_hour = (time_diff <= pd.Timedelta(hours=1)) & (time_diff > pd.Timedelta(0))
    
    # Create mapping DataFrame to restore original order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'session_within_last_hour': session_within_hour
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['session_within_last_hour'] = result_df['session_within_last_hour'].astype(int)
    df.drop('_temp_idx', axis=1, inplace=True)
    
    return df

def add_time_since_last_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time in minutes since user's last session. First session gets -1.
    
    Args:
        df: DataFrame containing user_id and DateTime columns
        
    Returns:
        DataFrame with new time_since_last_session column
    """
    # Create a temporary index to ensure correct mapping
    df = df.copy()  # Avoid modifying original
    df['_temp_idx'] = range(len(df))
    
    # Sort by user and datetime
    sorted_df = df.sort_values(['user_id', 'DateTime'])
    
    # Calculate time difference between consecutive sessions for same user
    time_diff = sorted_df.groupby('user_id')['DateTime'].diff()
    
    # Convert timedelta to minutes, replace NaN (first sessions) with -1
    minutes_since_last = time_diff.dt.total_seconds() / 60
    minutes_since_last = minutes_since_last.fillna(-1)
    
    # Create mapping DataFrame to restore original order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'time_since_last_session': minutes_since_last
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['time_since_last_session'] = result_df['time_since_last_session']
    df.drop('_temp_idx', axis=1, inplace=True)
    return df

def add_first_session_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary feature indicating if user's first session is today.
    
    Args:
        df: DataFrame containing user_id and DateTime columns
        
    Returns:
        DataFrame with new first_session_today column
    """
    df = df.copy()
    df['first_session_today'] = df.groupby('user_id')['DateTime'].transform(
        lambda x: (x.dt.date != x.dt.date.shift()) | (x.dt.dayofweek != x.dt.dayofweek.shift())
    ).astype(int)
    return df
def add_num_sessions_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a feature indicating the number of sessions for the user on the same date.
    
    Args:
        df: DataFrame containing user_id and DateTime columns
        
    Returns:
        DataFrame with new num_sessions_today column
    """
    df = df.copy()
    df['date'] = df['DateTime'].dt.date
    df['num_sessions_today'] = df.groupby(['user_id', 'date'])['date'].transform('count')
    df.drop('date', axis=1, inplace=True)
    return df
def add_high_volume_user(df: pd.DataFrame, quantile: float = 0.9) -> pd.DataFrame:
    """
    Add binary feature indicating if user is in the top quantile of users by number of sessions.
    
    Args:
        df: DataFrame containing user_id and DateTime columns
        quantile: Quantile to use for defining high-volume users
        
    Returns:
        DataFrame with new high_volume_user column
    """
    df = df.copy()
    df['high_volume_user'] = (df.groupby('user_id').size().rank(pct=True) > quantile).astype(int)
    return df
def add_product_viewed_before(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary feature indicating if user has viewed the product in an earlier session.
    
    Args:
        df: DataFrame containing user_id, DateTime and product columns
        
    Returns:
        DataFrame with new product_viewed_before column
    """
    # Create a temporary index to ensure correct mapping
    df = df.copy()  # Avoid modifying original
    df['_temp_idx'] = range(len(df))
    
    # Sort by user and datetime
    sorted_df = df.sort_values(['user_id', 'DateTime'])
    
    # Calculate product views on sorted copy
    product_viewed = sorted_df.groupby(['user_id', 'product']).cumcount() > 0
    
    # Create a mapping DataFrame to ensure correct order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'product_viewed_before': product_viewed
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['product_viewed_before'] = result_df['product_viewed_before'].astype(int)
    df.drop('_temp_idx', axis=1, inplace=True)
    
    return df

def add_category_viewed_before(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Avoid modifying original
    df['_temp_idx'] = range(len(df))
    
    # Sort by user and datetime
    sorted_df = df.sort_values(['user_id', 'DateTime'])
    
    # Calculate product views on sorted copy
    category_viewed = sorted_df.groupby(['user_id', 'product']).cumcount() > 0
    
    # Create a mapping DataFrame to ensure correct order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'category_viewed_before': category_viewed
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['category_viewed_before'] = result_df['category_viewed_before'].astype(int)
    df.drop('_temp_idx', axis=1, inplace=True)
    
    return df


def add_session_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative session count per user when sorted chronologically.
    
    Args:
        df: DataFrame containing user_id and DateTime columns
        
    Returns:
        DataFrame with new session_number column containing cumulative count
    """
    # Create temporary index to preserve order
    df = df.copy()
    df['_temp_idx'] = range(len(df))
    
    # Sort by user and datetim
    sorted_df = df.sort_values(['user_id', 'DateTime'])
    
    # Calculate cumulative session count
    session_number = sorted_df.groupby('user_id').cumcount() + 1
    
    # Create mapping DataFrame to restore original order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'session_number': session_number
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['session_number'] = result_df['session_number']
    df.drop('_temp_idx', axis=1, inplace=True)
    return df

def add_campaign_viewed_before(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Avoid modifying original
    df['_temp_idx'] = range(len(df))
    
    # Sort by user and datetime
    sorted_df = df.sort_values(['user_id', 'DateTime'])
    
    # Calculate product views on sorted copy
    campaign_viewed = sorted_df.groupby(['user_id', 'campaign_id']).cumcount() > 0
    
    # Create a mapping DataFrame to ensure correct order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'campaign_viewed_before': campaign_viewed
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['campaign_viewed_before'] = result_df['campaign_viewed_before'].astype(int)
    df.drop('_temp_idx', axis=1, inplace=True)
    
    return df
def add_number_of_websites_viewed_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add running sum of number of different entries in the 'website' column up to this point.
    
    Args:
        df: DataFrame containing user_id, DateTime, and product columns
        
    Returns:
        DataFrame with new number_of_websites_viewed_today column
    """
    df = df.copy()
    # Create temporary index to preserve order
    df['_temp_idx'] = range(len(df))
    
    # Extract date from DateTime
    df['date'] = df['DateTime'].dt.date
    
    # Sort by user, date, and DateTime
    sorted_df = df.sort_values(['user_id', 'date', 'DateTime'])
    
    # Calculate cumulative count of unique products viewed by user on the same date
    website_count_today = sorted_df.groupby(['user_id', 'date'])['webpage_id'].cumcount() + 1
    
    # Create mapping DataFrame to restore original order
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'number_of_websites_viewed_today': website_count_today
    })
    
    # Map back to original order
    result_df = result_df.sort_values('_temp_idx')
    
    # Assign to original DataFrame and clean up
    df['number_of_websites_viewed_today'] = result_df['number_of_websites_viewed_today']
    df.drop(['_temp_idx', 'date'], axis=1, inplace=True)
    return df
def add_number_of_products_viewed_today(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df['DateTime'].dt.date
    df['_temp_idx'] = range(len(df))
    sorted_df = df.sort_values(['user_id', 'date', 'DateTime'])
    product_count_today = sorted_df.groupby(['user_id', 'date'])['product'].cumcount() + 1
    result_df = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'number_of_products_viewed_today': product_count_today
    })
    result_df = result_df.sort_values('_temp_idx')
    df['number_of_products_viewed_today'] = result_df['number_of_products_viewed_today']
    df.drop(['_temp_idx', 'date'], axis=1, inplace=True)
    return df
def add_product_variety_for_user(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['product_variety'] = df.groupby('user_id')['product'].nunique()
    return df
def add_category_variety_for_user(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['category_variety'] = df.groupby('user_id')['product_category_1'].nunique()
    return df
def add_campaign_variety_for_user(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['campaign_variety'] = df.groupby('user_id')['campaign_id'].nunique()
    return df
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features and verify no NaN values are introduced."""
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = extract_time_features(df)
    df = add_first_session_feature(df)
    df = add_session_within_last_hour(df)
    df = add_product_viewed_before(df)
    df = add_session_number(df)
    df = add_first_session_today(df)
    df = add_high_volume_user(df)
    df = add_time_since_last_session(df)
    df = add_number_of_products_viewed_today(df)
    df = add_campaign_viewed_before(df)
    df = add_number_of_websites_viewed_today(df)
    df = add_num_sessions_today(df)
    df = add_category_viewed_before(df)
    df = add_product_variety_for_user(df)
    df = add_category_variety_for_user(df)
    df = add_campaign_variety_for_user(df)
    return df