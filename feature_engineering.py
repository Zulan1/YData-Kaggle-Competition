import pandas as pd
import config as conf
from functools import partial
from feature_registry import Feature, FeatureRegistry
import pandas as pd

# Constant with original features.
ORIGINAL_FEATURES = [
    'product',
    'product_category_1',
    'campaign_id',
    'webpage_id',
    'user_group_id',
    'gender',
    'var_1',
    'user_depth'
]

# Configuration for each original feature.
# You can easily adjust the scope and categorical flag here.
ORIGINAL_FEATURES_CONFIG = {
    'product': {'scope': 'session', 'categorical': True},
    'product_category_1': {'scope': 'session', 'categorical': True},
    'campaign_id': {'scope': 'session', 'categorical': True},
    'webpage_id': {'scope': 'session', 'categorical': True},
    'user_group_id': {'scope': 'user', 'categorical': False},
    'gender': {'scope': 'user', 'categorical': True},
    'var_1': {'scope': 'session', 'categorical': False},
    'user_depth': {'scope': 'user', 'categorical': False},
}



###############################################################################
#                       Top-level helper functions (for picklability)
###############################################################################

# Original features.
def original_feature_func(df: pd.DataFrame, col: str):
    return df[col]

def make_original_feature_func(col: str):
    """
    Returns a picklable function (using partial) that gets the original column.
    """
    return partial(original_feature_func, col=col)


# For variety features.
def variety_for_user(df: pd.DataFrame, column: str) -> pd.Series:
    series = df.groupby('user_id')[column].transform('nunique')
    series.name = f'{column}_variety'
    return series

def variety_for_user_func(column: str):
    return partial(variety_for_user, column=column)


# For most common category features.
def most_common_category_for_user(df: pd.DataFrame, column: str) -> pd.Series:
    product_stats = (df.groupby(['user_id', column])
                     .agg({'DateTime': 'max', column: 'size'})
                     .rename(columns={column: 'count'}))
    most_common = (product_stats.reset_index()
                   .sort_values(['user_id', 'count', 'DateTime'],
                                ascending=[True, False, False])
                   .groupby('user_id')
                   .first()
                   .reset_index()[['user_id', column]])
    series = df['user_id'].map(most_common.set_index('user_id')[column])
    series.name = f'most_common_{column}'
    return series

def most_common_category_for_user_func(column: str):
    return partial(most_common_category_for_user, column=column)


# For first session today feature.
def first_session_today_transform(x):
    return ((x.dt.date != x.dt.date.shift()) | (x.dt.dayofweek != x.dt.dayofweek.shift())).astype(int)

def first_session_today_func(df: pd.DataFrame) -> pd.Series:
    series = df.groupby('user_id')['DateTime'].transform(first_session_today_transform)
    series.name = 'first_session_today'
    return series

###############################################################################
#                       Engineered feature functions
#
# (The remaining feature functions are defined at the module-level.)
###############################################################################

def first_session_feature_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a binary Series indicating if the session is the first for the user.
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    first_session = ~sorted_df['user_id'].duplicated()
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'first_session': first_session
    }).sort_values('_temp_idx')
    series = result['first_session'].astype(int)
    series.index = df.index
    return series


def extract_time_features_func(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time features from the DateTime column.
    Returns one-hot day of week and several binary time-range features.
    """
    new_features = {}
    days = df['DateTime'].dt.dayofweek
    for day in range(7):
        new_features[f'day_{day}'] = (days == day).astype(int)
    
    hours = df['DateTime'].dt.hour
    new_features['18_21'] = ((hours >= 18) & (hours < 21)).astype(int)
    new_features['21_00'] = ((hours >= 21) | (hours < 0)).astype(int)
    new_features['00_08'] = ((hours >= 0) & (hours < 8)).astype(int)
    new_features['08_13'] = ((hours >= 8) & (hours < 13)).astype(int)
    new_features['13_18'] = ((hours >= 13) & (hours < 18)).astype(int)
    
    return pd.DataFrame(new_features, index=df.index)


def session_within_last_hour_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a binary Series indicating if the user had another session 
    within the last hour.
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    time_diff = sorted_df.groupby('user_id')['DateTime'].diff()
    session_within = (time_diff <= pd.Timedelta(hours=1)) & (time_diff > pd.Timedelta(0))
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'session_within_last_hour': session_within
    }).sort_values('_temp_idx')
    series = result['session_within_last_hour'].fillna(False).astype(int)
    series.index = df.index
    return series


def time_since_last_session_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with the time in minutes since the user's last session 
    (first session gets -1).
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    time_diff = sorted_df.groupby('user_id')['DateTime'].diff()
    minutes_since_last = (time_diff.dt.total_seconds() / 60).fillna(-1)
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'time_since_last_session': minutes_since_last
    }).sort_values('_temp_idx')
    series = result['time_since_last_session'].astype(int)
    series.index = df.index
    return series


def num_sessions_today_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indicating the number of sessions the user has on the same day.
    """
    temp = df.copy()
    temp['date'] = temp['DateTime'].dt.date
    series = temp.groupby(['user_id', 'date'])['date'].transform('count')
    series.name = 'num_sessions_today'
    return series


def high_volume_user_func(df: pd.DataFrame, quantile: float = 0.95) -> pd.Series:
    """
    Returns a binary Series indicating if the user is among the top quantile by number of sessions.
    """
    user_session_counts = df.groupby('user_id').size()
    high_volume_threshold = user_session_counts.quantile(quantile)
    high_volume_users = user_session_counts[user_session_counts >= high_volume_threshold].index
    series = df['user_id'].isin(high_volume_users).astype(int)
    series.name = 'high_volume_user'
    return series


def product_viewed_before_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a binary Series indicating if a product has been viewed by the user before.
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    product_viewed = sorted_df.groupby(['user_id', 'product']).cumcount() > 0
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'product_viewed_before': product_viewed
    }).sort_values('_temp_idx')
    series = result['product_viewed_before'].astype(int)
    series.index = df.index
    return series


def category_viewed_before_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a binary Series indicating if a product category has been viewed 
    by the user before. (Uses the product_category_1 column.)
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    category_viewed = sorted_df.groupby(['user_id', 'product_category_1']).cumcount() > 0
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'category_viewed_before': category_viewed
    }).sort_values('_temp_idx')
    series = result['category_viewed_before'].astype(int)
    series.index = df.index
    return series


def session_number_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with the cumulative session count per user.
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    session_number = sorted_df.groupby('user_id').cumcount() + 1
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'session_number': session_number
    }).sort_values('_temp_idx')
    series = result['session_number']
    series.index = df.index
    return series


def campaign_viewed_before_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a binary Series indicating if a campaign has been viewed by the user before.
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    campaign_viewed = sorted_df.groupby(['user_id', 'campaign_id']).cumcount() > 0
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'campaign_viewed_before': campaign_viewed
    }).sort_values('_temp_idx')
    series = result['campaign_viewed_before'].astype(int)
    series.index = df.index
    return series


def number_of_websites_viewed_today_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with the cumulative count of websites viewed by a user today.
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    temp['date'] = temp['DateTime'].dt.date
    sorted_df = temp.sort_values(['user_id', 'date', 'DateTime'])
    website_count_today = sorted_df.groupby(['user_id', 'date'])['webpage_id'].cumcount() + 1
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'number_of_websites_viewed_today': website_count_today
    }).sort_values('_temp_idx')
    series = result['number_of_websites_viewed_today']
    series.index = df.index
    return series

def add_total_number_of_sessions_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with the cumulative count of sessions for a user.
    """
    series = df.groupby('user_id')['session_id'].cumcount() + 1
    series.name = 'total_number_of_sessions'
    return series

def number_of_products_viewed_today_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with the cumulative count of products viewed by a user today.
    """
    temp = df.copy()
    temp['date'] = temp['DateTime'].dt.date
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'date', 'DateTime'])
    product_count_today = sorted_df.groupby(['user_id', 'date'])['product'].cumcount() + 1
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'number_of_products_viewed_today': product_count_today
    }).sort_values('_temp_idx')
    series = result['number_of_products_viewed_today']
    series.index = df.index
    return series


def same_product_as_previous_session_func(df: pd.DataFrame) -> pd.Series:
    """
    Returns a binary Series indicating if the product in the current session
    is the same as that in the previous session for this user.
    """
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    same_product = sorted_df.groupby('user_id')['product'].shift() == sorted_df['product']
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'same_product_as_previous_session': same_product
    }).sort_values('_temp_idx')
    series = result['same_product_as_previous_session'].fillna(False).astype(int)
    series.index = df.index
    return series

def add_same_website_as_previous_session_func(df: pd.DataFrame) -> pd.Series:
    temp = df.copy()
    temp['_temp_idx'] = temp.index
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    same_product = sorted_df.groupby('user_id')['webpage_id'].shift() == sorted_df['webpage_id']
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        'same_website_as_previous_session': same_product
    }).sort_values('_temp_idx')
    series = result['same_website_as_previous_session'].fillna(False).astype(int)
    series.index = df.index
    return series

def num_times_viewed_col_func(df: pd.DataFrame, col: str) -> pd.Series:
    temp = df.copy()
    temp['_temp_idx'] = temp.index
        # Sort to ensure chronological order per user for cumulative counting.
    sorted_df = temp.sort_values(['user_id', 'DateTime'])
    cum_count = sorted_df.groupby(['user_id', col]).cumcount() + 1
    result = pd.DataFrame({
        '_temp_idx': sorted_df['_temp_idx'],
        f'num_times_viewed_{col}': cum_count
        }).sort_values('_temp_idx')
    series = result[f'num_times_viewed_{col}']
    series.index = df.index
    return series

def add_num_times_viewed_func(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['product', 'campaign_id', 'webpage_id', 'product_category_1']:
        df[f'num_times_viewed_{col}'] = num_times_viewed_col_func(df, col)
    return df

def add_most_common_category_features_func(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['product', 'campaign_id', 'webpage_id', 'product_category_1']:
        mode_per_user = df.groupby('user_id')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        df[f'most_common_{col}'] = df['user_id'].map(mode_per_user).astype(str)
    return df

###############################################################################
#                       add_features function
#
# This function converts the DateTime column, creates and registers a Feature
# for every engineered and original feature, and then applies only the engineered features.
# Original features are registered for introspection but not added to the DataFrame.
###############################################################################

def add_features(df: pd.DataFrame, add_catboost_features: bool = conf.USE_CATBOOST) -> (pd.DataFrame, FeatureRegistry):
    """
    Adds all engineered features (and leaves through original features untouched) using the FeatureRegistry.
    Returns a tuple (df_transformed, registry) where df_transformed is the modified DataFrame,
    and registry is a FeatureRegistry of all registered features.
    """
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    
    registry = FeatureRegistry()

    # Register original features (do not re-add to DataFrame).
    for col in ORIGINAL_FEATURES:
        cfg = ORIGINAL_FEATURES_CONFIG.get(col, {'scope': 'session', 'categorical': False})
        registry.register(Feature(
            name=col,
            func=make_original_feature_func(col),
            scope=cfg['scope'],
            categorical=cfg['categorical'],
            add_to_df=False
        ))

    # Register engineered features.
    registry.register(Feature(
        name="first_session",
        func=first_session_feature_func,
        scope="session",
        categorical=True
    ))

    registry.register(Feature(
        name="total_number_of_sessions",
        func=add_total_number_of_sessions_func,
        scope="user",
        categorical=False
    ))
    
    # extract_time_features returns multiple columns.
    registry.register(Feature(
        name="time_features",
        func=extract_time_features_func,
        scope="session",
        categorical=True
    ))
    
    registry.register(Feature(
        name="session_within_last_hour",
        func=session_within_last_hour_func,
        scope="session",
        categorical=True
    ))
    
    registry.register(Feature(
        name="time_since_last_session",
        func=time_since_last_session_func,
        scope="session",
        categorical=False
    ))
    
    registry.register(Feature(
        name="first_session_today",
        func=first_session_today_func,
        scope="session",
        categorical=True
    ))
    
    registry.register(Feature(
        name="num_sessions_today",
        func=num_sessions_today_func,
        scope="session",
        categorical=False
    ))
    
    # Use partial instead of a lambda for high_volume_user.
    registry.register(Feature(
        name="high_volume_user",
        func=partial(high_volume_user_func, quantile=0.95),
        scope="user",
        categorical=True
    ))
    
    registry.register(Feature(
        name="product_viewed_before",
        func=product_viewed_before_func,
        scope="session",
        categorical=True
    ))
    
    registry.register(Feature(
        name="category_viewed_before",
        func=category_viewed_before_func,
        scope="session",
        categorical=True
    ))
    
    registry.register(Feature(
        name="session_number",
        func=session_number_func,
        scope="session",
        categorical=False
    ))
    
    registry.register(Feature(
        name="campaign_viewed_before",
        func=campaign_viewed_before_func,
        scope="session",
        categorical=True
    ))
    
    registry.register(Feature(
        name="number_of_websites_viewed_today",
        func=number_of_websites_viewed_today_func,
        scope="session",
        categorical=False
    ))
    
    registry.register(Feature(
        name="number_of_products_viewed_today",
        func=number_of_products_viewed_today_func,
        scope="session",
        categorical=False
    ))
    
    # Variety features.
    registry.register(Feature(
        name="product_variety",
        func=variety_for_user_func('product'),
        scope="user",
        categorical=False
    ))
    registry.register(Feature(
        name="product_category_1_variety",
        func=variety_for_user_func('product_category_1'),
        scope="user",
        categorical=False
    ))
    registry.register(Feature(
        name="campaign_id_variety",
        func=variety_for_user_func('campaign_id'),
        scope="user",
        categorical=False
    ))
    registry.register(Feature(
        name="webpage_id_variety",
        func=variety_for_user_func('webpage_id'),
        scope="user",
        categorical=False
    ))
    
    registry.register(Feature(
        name="same_product_as_previous_session",
        func=same_product_as_previous_session_func,
        scope="session",
        categorical=True
    ))

    registry.register(Feature(
        name="most_common_category_features",
        func=add_most_common_category_features_func,
        scope="user",
        categorical=True
    ))
    
    # Only engineered features (those flagged add_to_df=True) are applied.
    df_transformed = registry.transform(df)
    return df_transformed, registry
