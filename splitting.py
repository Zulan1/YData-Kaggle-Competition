import config as conf
import numpy as np
import pandas as pd
import splitting_constants

class DataSplitter:
    """
    A class to split the dataset into training, validation, and testing sets according to
    specific rules based on user behavior and configurable criteria.

    Attributes:
        split_ratios (tuple of float): Fractions for splitting data into (train, validation, test). Must sum to 1.
        random_state (int): Seed used for reproducibility.
        test_product (any): The product that may be withheld from the training data.
        leave_one_product_out (bool): If True, removes sessions with the test product from training.
        verbose (bool): If True, prints detailed splitting statistics.
    """
    
    def __init__(self, 
                 split_ratios=splitting_constants.TRAIN_TEST_VAL_SPLIT, 
                 random_state=conf.RANDOM_STATE,
                 test_product=splitting_constants.PRODUCT_TO_LEAVE, 
                 leave_one_product_out=splitting_constants.DEFAULT_LEAVE_ONE_PRODUCT_OUT,
                 verbose=False):
        self.split_ratios = split_ratios
        self.random_state = random_state
        self.test_product = test_product
        self.leave_one_product_out = leave_one_product_out
        self.verbose = verbose
        
    def split_data(self, df: pd.DataFrame):
        """
        Splits the DataFrame into training, validation, and testing sets.
        
        Splitting Logic:
          - Identify users who have interacted with the test product ("rare" users) and those who haven't ("normal" users).
          - Split each group into training, validation, and testing sets using the split_by_user method.
          - For rare users in the training set, optionally remove sessions involving the test product.
          - Optionally remove rows corresponding to the test user group from the training set.
        
        Prints splitting statistics if verbose is enabled.
        
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Identify rare and normal users based on test_product
        rare_user_ids = set(df.loc[df['product'] == self.test_product, 'user_id'])
        normal_user_ids = set(df['user_id']) - rare_user_ids
        
        # Partition the DataFrame
        df_normal = df[df['user_id'].isin(normal_user_ids)]
        df_rare = df[df['user_id'].isin(rare_user_ids)]
        
        # Split normal users into train, validation, and test sets
        normal_train, normal_val, normal_test = self.split_by_user(df_normal)
        
        # Split rare users into train, validation, and test sets
        rare_train_all, rare_val, rare_test = self.split_by_user(df_rare)
        if self.leave_one_product_out:
            rare_train = rare_train_all[rare_train_all['product'] != self.test_product]
        else:
            rare_train = rare_train_all
        
        # Combine splits from normal and rare users
        df_train = pd.concat([normal_train, rare_train]).sort_index()
        df_val = pd.concat([normal_val, rare_val]).sort_index()
        df_test = pd.concat([normal_test, rare_test]).sort_index()
        
        # Calculate summary statistics for logging purposes.
        total_sessions = len(df)
        train_pct = (len(df_train) / total_sessions) * 100
        val_pct = (len(df_val) / total_sessions) * 100
        test_pct = (len(df_test) / total_sessions) * 100

        train_ctr = df_train['is_click'].mean() if len(df_train) > 0 else 0
        val_ctr = df_val['is_click'].mean() if len(df_val) > 0 else 0
        test_ctr = df_test['is_click'].mean() if len(df_test) > 0 else 0
        
        if self.verbose:
            print("\n[splitting.py] Splitting Statistics:")
            print(f"[splitting.py] Train sessions: {train_pct:.2f}% of total, CTR: {train_ctr:.4f}")
            print(f"[splitting.py] Validation sessions: {val_pct:.2f}% of total, CTR: {val_ctr:.4f}")
            print(f"[splitting.py] Test sessions: {test_pct:.2f}% of total, CTR: {test_ctr:.4f}")
            if self.leave_one_product_out:
                print("[splitting.py] Test product left out of training set: ", self.test_product)
        
        # Integrity check: enforce test product removal if configured.
        if self.leave_one_product_out:
            assert self.test_product not in df_train['product'].unique(), (
                f"df_train contains the test product: {self.test_product}"
            )
        
        return df_train, df_val, df_test
    
    def split_by_user(self, df: pd.DataFrame):
        """
        Splits the DataFrame into train, validation, and test subsets while attempting to balance user behavior.
        
        Process:
          - Aggregates user statistics (number of sessions and clicks).
          - Groups users by a derived "click_distribution" string.
          - Separates out groups with scarce representation (rare users) and then splits users within each group 
            based on the provided split_ratios.
        
        Returns:
            tuple: (train_split, val_split, test_split)
        """
        if not np.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("Split ratios must sum to 1")
        
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
        # Separate groups with few users ("rare") from those with many ("non-rare").
        for group_key, users in behavior_groups.items():
            if len(users) < rare_threshold:
                rare_users.extend(users)
            else:
                non_rare_groups[group_key] = users
        if rare_users:
            non_rare_groups["rare"] = rare_users
        behavior_groups = non_rare_groups
        
        first_users, second_users, third_users = [], [], []
        
        for users in behavior_groups.values():
            users = np.array(users)
            np.random.seed(self.random_state)
            np.random.shuffle(users)
            
            n_users = len(users)
            n_first = int(round(self.split_ratios[0] * n_users))
            n_second = int(round(self.split_ratios[1] * n_users))
            if n_first + n_second > n_users:
                n_second = n_users - n_first
            
            first_users.extend(users[:n_first])
            second_users.extend(users[n_first:n_first + n_second])
            third_users.extend(users[n_first + n_second:])
        
        first_split = df[df['user_id'].isin(first_users)]
        second_split = df[df['user_id'].isin(second_users)]
        third_split = df[df['user_id'].isin(third_users)]
        
        return first_split, second_split, third_split