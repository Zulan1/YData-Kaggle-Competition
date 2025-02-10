import config as conf
import numpy as np

def split_by_user(df, split_ratios=conf.TRAIN_TEST_VAL_SPLIT):
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
    
    # Initialize user lists for each split
    first_users, second_users, third_users = [], [], []
    
    # Split users within each behavior group
    for users in behavior_groups.values():
        users = np.array(users)
        np.random.seed(conf.RANDOM_STATE)
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