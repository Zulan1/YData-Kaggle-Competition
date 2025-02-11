import config as conf
import numpy as np
import pandas as pd

def split_data(df, split_ratios=conf.TRAIN_TEST_VAL_SPLIT, random_state=conf.RANDOM_STATE,
                          test_product=conf.PRODUCT_TO_LEAVE, verbose=False):
    """
    Performs a user-based split while ensuring product 'A' is unseen in training.

    Strategy:
      1. Split users who NEVER interacted with "A" (60-20-20).
      2. Split users who interacted with "A" (60-20-20):
         - In train: Keep users but remove all 'A' sessions.
         - In val/test: Keep users with all sessions, including 'A'.
    
    Parameters:
      df           : DataFrame with the dataset.
      product_col  : Column containing product categories.
      unseen_product : The specific product to be removed from training.
      user_col     : Column that uniquely identifies users.
      train_frac, val_frac, test_frac : Fractions for dataset split.
      random_state : Seed for reproducibility.

    Returns:
      train_df, val_df, test_df: DataFrames split accordingly.
    """
    # Split users into rare and non-rare based on interaction with the test product
    rare_user_ids = set(df.loc[df['product'] == test_product, 'user_id'])
    normal_user_ids = set(df['user_id']) - rare_user_ids

    # Partition the DataFrame into normal users (who never saw the unseen product)
    # and rare users (who have at least one session with the unseen product)
    df_normal = df[df['user_id'].isin(normal_user_ids)]
    df_rare = df[df['user_id'].isin(rare_user_ids)]

    # Split normal users into train, validation, and test
    normal_train, normal_val, normal_test = split_by_user(df_normal, split_ratios)

    # Split rare users into train, validation, and test
    rare_train_all, rare_val, rare_test = split_by_user(df_rare, split_ratios)
    # For rare users, remove sessions with the unseen product from the training split
    rare_train = rare_train_all[rare_train_all['product'] != test_product]

    # Combine the corresponding splits from normal and rare users
    df_train = pd.concat([normal_train, rare_train]).sort_index()
    df_val = pd.concat([normal_val, rare_val]).sort_index()
    df_test = pd.concat([normal_test, rare_test]).sort_index()

    # Compute session proportions and mean CTR for each split
    total_sessions = len(df)
    train_pct = (len(df_train) / total_sessions) * 100
    val_pct = (len(df_val) / total_sessions) * 100
    test_pct = (len(df_test) / total_sessions) * 100

    train_ctr = df_train['is_click'].mean() if len(df_train) > 0 else 0
    val_ctr = df_val['is_click'].mean() if len(df_val) > 0 else 0
    test_ctr = df_test['is_click'].mean() if len(df_test) > 0 else 0

    if verbose:

        print(f"Train sessions: {train_pct:.2f}% of total, CTR: {train_ctr:.4f}")
        print(f"Validation sessions: {val_pct:.2f}% of total, CTR: {val_ctr:.4f}")
        print(f"Test sessions: {test_pct:.2f}% of total, CTR: {test_ctr:.4f}")

    assert test_product not in df_train['product'].unique(), f"df_train contains the product to leave: {test_product}"

    return df_train, df_val, df_test



def split_by_user(df, split_ratios=conf.TRAIN_TEST_VAL_SPLIT, random_state=conf.RANDOM_STATE):
    """Split DataFrame into train/val/test while keeping user behavior balanced."""
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")
    
    # Get user stats and behavior categories
    user_stats = df.groupby('user_id').agg({
        'session_id': 'nunique',
        'is_click': 'sum'
    }).reset_index()
    
    user_stats['click_distribution'] = user_stats.apply(
        lambda x: f"{x['session_id']}_sessions_{x['is_click']}_clicks", axis=1
    )
    
    behavior_groups = user_stats.groupby('click_distribution')['user_id'].apply(list).to_dict()

    rare_threshold = 20
    rare_users = []
    non_rare_groups = {}
    for group_key, users in behavior_groups.items():
        if len(users) < rare_threshold:
            rare_users.extend(users)
        else:
            non_rare_groups[group_key] = users
    if rare_users:
        non_rare_groups["rare"] = rare_users
    behavior_groups = non_rare_groups
    
    # Initialize user lists for each split
    first_users, second_users, third_users = [], [], []
    
    # Split users within each behavior group
    for users in behavior_groups.values():
        users = np.array(users)
        np.random.seed(random_state)
        np.random.shuffle(users)
        
        n_users = len(users)
        n_first = int(round(split_ratios[0] * n_users))
        n_second = int(round(split_ratios[1] * n_users))
        
        if n_first + n_second > n_users:
            n_second = n_users - n_first
            
        first_users.extend(users[:n_first])
        second_users.extend(users[n_first:n_first + n_second])
        third_users.extend(users[n_first + n_second:])
    
    # Create splits based on user assignments
    first_split = df[df['user_id'].isin(first_users)]
    second_split = df[df['user_id'].isin(second_users)]
    third_split = df[df['user_id'].isin(third_users)]
    
    return first_split, second_split, third_split