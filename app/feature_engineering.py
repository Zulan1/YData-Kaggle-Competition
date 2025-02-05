import pandas as pd

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
