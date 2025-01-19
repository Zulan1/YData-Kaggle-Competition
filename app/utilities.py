#definition of utility functions:
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
#-------------------------------------------------------
#                       General
#-------------------------------------------------------
def display_unique(df):
    '''Loop through the columns and print the levels of each categorial variable'''
    for j,col in enumerate(df.columns):
        if any(df[col].isna()):
          nan_str = f'[nans = {df[col].isna().sum()}]'
        else:
          nan_str = ''
        unique_vals = df[col].dropna().unique()
        L = len(unique_vals)
        if len(unique_vals) <= 10:
            print(f"{j+1}. {col} (n={L}): {nan_str} {unique_vals}")
        else:
            print(f"{j+1}. {col} (n={L}):{nan_str} First 10 values: {unique_vals[0:9]}")

#-------------------------------------------------------
#                       Part 1
#-------------------------------------------------------
# Custom function to categorize the values
def categorize(value):
    if value == 0:
        return 'complete'
    elif 0 < value < 1:
        return 'partial'
    elif value == 1:
        return 'absent'
    else:
        return 'other'

def categorize_series (x : pd.Series) -> pd.Series:
  # Apply the custom function and count the occurrences
  categories = x.apply(categorize)
  result = categories.value_counts()

  # Convert the result to a Series with a fixed index order
  result = result.reindex(['equal_0', 'between_0_and_1', 'equal_1'], fill_value=0)

  return result

from typing import Optional
def watch_user (df, user_id, columns : Optional[list] = None) -> pd.DataFrame:
  """ filter all of the records of a certain user, with optional to subset columns

  Args:
    df - pandas dataframe
    user_id (float) user id for which the column 'user_id' equals
    columns (list, optional): list of columns to subset

  Returns:
    sliced pandas dataframe
  """
  if columns is None:
    out = df.loc[df.user_id == user_id,:]
  else:
    out = df.loc[df.user_id == user_id, columns]
  return out

def convert_float_vars_to_ints (df):
    """Convert float with no nans columns to integers for better visualization
    """
    df = df.copy()
    for var in df.columns:
      if ((df[var].dtypes == float) and (not df[var].isna().any())):
        df[var] = df[var].astype(int)

    return df


def compute_missing_info (df, features):
  """ compute data structures required for the operation of impute_users_data().
   The code is divided between the two function to let the user execute them seperately, as the computation of missing data takes long time.

   Args:
          df: original dataframe of the CTR database
          features: list of *demographic* features

   Returns:
        missing_info: a dataframe containing categorization of the information completeness (complete, partial, absent)
        missing_data_by_user: the same data frame with numeric values instad
  """
  #group by user_id:
  grouped_by_users = df.groupby(['user_id'])[features]

  #comppute mean rate of missing data:
  missing_data_by_user = grouped_by_users.apply(lambda x: x.isna().mean())

  #categorize data
  binned_data = list()
  for feature in features:
    binned_data.append(missing_data_by_user[feature].apply(categorize))

  #create a dataframe containing missing info:
  missing_info = pd.concat(binned_data, axis=1)

  return missing_info, missing_data_by_user


def impute_users_data (df, features, missing_info, missing_data_by_user):
  """Impute demographic data (of users)

  Args:
          df: original dataframe of the CTR database
          features: list of *demographic* features
          missing_info: a dataframe containing categorization of the information completeness (complete, partial, absent) - created by compute_missing_info()
          missing_data_by_user: the same data frame with numeric values instad - created by compute_missing_info()

  Returns:
          imputed_df: the same data frame with imputed values. Missing data in 'features' column is not removed!
          n_imputed: series with index=features that counts how many values were imputed for each variable

  """

  #initialize a imputed vesion of the df:
  df_imputed = df.copy()

  #this code uses iteration, which inefficient, but for now it works. vectorize it later.
  #for each user
  n_impute = pd.Series(np.zeros((len(features)),),index=features)
  for user in missing_info.index:
    #and feature
    for j,feature in enumerate(features):
      #check if data is indeed partial (for other cases we have nothing to do)
      if missing_info.loc[user, feature]=='partial':
        #1. retrieve the data
        user_df = watch_user(df_imputed, user, [feature])

        #2. drop nans
        values = user_df[feature].dropna().to_list()

        #4. take the first, assume that all of the rest is equal.
        #if not, it is a real problem and means there is disagreement in the data
        #for now we leave this
        impute = values[0]
        n_impute[feature] += 1


        #5. find all of the places were there are missing data in this place:
        df_imputed.loc[((df_imputed.user_id == user) & df_imputed[feature].isna()),feature] = impute

  #return the imputed dataframe
  print (f'No. of imputed users: {n_impute}')
  return df_imputed, n_impute

def get_n_users (df, grouping_features):
  result = {}
  for feature in grouping_features:
    result[feature] = df.groupby([feature])['user_id'].nunique()
  return result

def compute_user_based_outcomes (df, features, filter_outliers):
  #filter_outliers is of form {feature_to_filter : [levels to filter]}
  click_rate_per_user = df.groupby(['user_id'])['is_click'].mean()
  sessions_per_user = df.groupby(['user_id'])['session_id'].count()
  users = pd.merge(sessions_per_user, click_rate_per_user, left_index=True, right_index=True, how='inner')
  users = users.rename(columns = {'session_id' : 'sessions_num', 'is_click' : 'click_rate'})
  users['click_num'] = users['click_rate'] * users['sessions_num']

  #slice the users that click at least once (a.k.a, clickers)
  clickers = users.loc[users['click_rate']>0,:]
  users['any_click'] = users['click_rate'].apply(lambda x : 1 if x > 0 else 0)

  Xy = {}
  outcomes = ['clickers_rate', 'click_rate', 'click_num', 'sessions_num']

  #create a dataset for each feature that includes this feature without nans
  for feature in features:
    xx = df.dropna(subset=[feature]).groupby(['user_id'])[feature].first()

    #Remove outlier groups:
    for feature_to_filter,levels_to_filter in filter_outliers.items():
      xx.filt = xx.copy()
      xx = xx.filt.loc[xx[feature_to_filter].isin(levels_to_filter),:]
    Xy[feature] = pd.merge(xx, users, left_index = True, right_index= True, how = 'inner').rename(columns={'any_click' : 'clickers_rate'})
    Xy[feature] = convert_float_vars_to_ints(Xy[feature])

  return Xy, outcomes

def plot_outcomes_and_features(data_dict, outcomes):
    """
    Plots multiple barplots as subplots from a dictionary of DataFrames.

    Args:
        data_dict (dict): A dictionary where keys are feature names and
                         values are DataFrames containing the feature and outcome column.
        outcome_col (str): The name of the outcome column present in all DataFrames.
    """


    num_cols = len(outcomes)  # Number of columns for subplots (adjust as needed)
    num_rows = len(data_dict)  # Calculate number of rows
    num_plots = num_cols * num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*2))
    for i, (feature_name, df) in enumerate(data_dict.items()):
      for j, outcome in enumerate(outcomes):
          ax = axes[i,j]
          sns.barplot(x=feature_name, y=outcome, data=df, ax=ax)
          #ax.set_title(f"{outcome} vs. {feature_name}")
          ax.set_xlabel(feature_name)
          ax.set_ylabel(outcome)

          # *** Improvement 1: Rotate xtick labels ***
          ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # # Hide any unused subplots
    # for j in range(num_plots, len(axes)):
    #     axes[j].set_visible(False)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def plot_outcomes_and_features2(data_dict, outcomes):
    """
    Plots multiple barplots as subplots from a dictionary of DataFrames.

    Args:
        data_dict (dict): A dictionary where keys are feature names and
                         values are DataFrames containing the feature and outcome column.
        outcome_col (str): The name of the outcome column present in all DataFrames.
    """


    num_cols = len(data_dict)
    num_rows = len(outcomes)
    num_plots = num_cols * num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    for i, (feature_name, df) in enumerate(data_dict.items()):
      for j, outcome in enumerate(outcomes):
          ax = axes[j,i]
          sns.barplot(x=feature_name, y=outcome, data=df, ax=ax)
          #ax.set_title(f"{outcome} vs. {feature_name}")
          ax.set_xlabel(feature_name)
          ax.set_ylabel(outcome)

          # *** Improvement 1: Rotate xtick labels after drawing ***
          plt.draw()  # Force the plot to draw and finalize tick positions
          ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

#-------------------------------------------------------
#                       Part 2
#-------------------------------------------------------
def combine_product_categories(df):
  """
  Combines product categories 1 and 2 into a new column 'product-combined'.

  Args:
    df: The input DataFrame containing 'product_category_1' and 'product_category_2' columns.

  Returns:
    A DataFrame with a new column 'product-combined' containing the combined categories.
  """

  # Create a copy of the DataFrame to avoid modifying the original
  df_combined = df.copy()

  # Fill NaN values with 0 in 'product_category_1' and 'product_category_2'
  df_combined[['product_category_1', 'product_category_2']] = df_combined[['product_category_1', 'product_category_2']].fillna(0)

  # Create the 'product-combined' column
  df_combined['product_cat_comb'] = df_combined.apply(lambda row: f"{int(row['product_category_1'])}_{int(row['product_category_2'])}", axis=1)

  return df_combined

def outcome_by_group_features (dataf, group_features, outcome_type = 'click_rate', min_sessions = 0):
  """
  Compute outcomes for grouped geatures

  Args:
    dataf: dataframe after creation of combined product category column.
    group_features: a list of features to group by. Currently full support is to either 1 or 2 features.
    outcome_type: measure to compute for each grouped data segment
      could be one of the following:
        - 'click_rate' (ratio between clicks and total sessions)
        - 'click_num' (absolute number of clicks)
        - 'sessions_num' (number of sessions)
    min_sessions: the minimal number of sessions that a grouped segments must have in order to be included in result


  Returns:
    outcome - a dataframe that contains the following columns:
      * Index (could be more the one), which contains the grouped_features
      * outcome column
      * session_number column (this column is produced anyway, even if outcome='sessions_num')
    map - a pandas table in which the rows correspond to the first grouped_features, and columns to the second, the hue to the outcome

  """


  df = dataf.copy()
  df.dropna(subset=['is_click'], inplace=True)

  xx = df.dropna(subset=group_features)
  outcome = None
  map = None
  grouped = xx.groupby(group_features)['is_click']

  #create outcome variables:
  if outcome_type == 'click_rate':
    outcome = grouped.mean().to_frame().rename(columns={'is_click':outcome_type})
  elif outcome_type == 'click_num':
    outcome = grouped.sum().to_frame().rename(columns={'is_click':outcome_type})
  elif outcome_type == 'sessions_num':
    outcome = grouped.count().to_frame().rename(columns={'is_click':outcome_type})

  if not outcome_type is None:
    #filter pairs with less than minimal number of sessions:
    sessions_num = grouped.count().to_frame().rename(columns={'is_click':'sessions_number'})
    outcome_sessions = pd.merge(outcome, sessions_num, left_index = True, right_index = True, how = 'inner')
    outcome_sessions = outcome_sessions.loc[outcome_sessions['sessions_number'] >= min_sessions,:]
    outcome = outcome_sessions

    #create maps:
    if len(group_features) == 2:
      map = outcome.pivot_table(index=group_features[0], columns = group_features[1], values = outcome_type).fillna(0)
    else:
      map = outcome.pivot_table(index=group_features[0], values = outcome_type).fillna(0)

  return outcome, map

def compute_and_plot_outcomes_for_product_campaign_pairs (df, product_features, outcomes, min_sessions = 0):
  """
  Compute and plot outcomes for product-campaing-category pairs.
  This a wrapper that allows for batch compuation and plotting of several outcomes and product-category features.


  Args:
    df: dataframe after creation of combined product category column.
    product_features: a list of features to group by. Currently full support is to either 1 or 2 features.
    outcomes: a list of measures to compute for each grouped data segment
      could be one of the following:
        - 'click_rate' (ratio between clicks and total sessions)
        - 'click_num' (absolute number of clicks)
        - 'sessions_num' (number of sessions)
    min_sessions: the minimal number of sessions that a grouped segments must have in order to be included in result


  Returns:
    * outcomes_df - a nested dictionary of outcomes, cotaining a dictionary of product features.
    The inner values are dataframes that contain the following columns:
      * Index (could be more the one), which contains the grouped_features
      * outcome column
      * session_number column (this column is produced anyway, even if outcome='sessions_num')

    * maps- a dictionary of the same structure like outcomes_df with pandas tables  in which the rows correspond to the first grouped_features, and columns to the second, the hue to the outcome

    * top10 - a dictionary of the same structure like outcomes_df, which containes the top 10 pairs with highest outcomes.

  """
  outcomes_df = {}
  maps = {}
  top10 = {}
  for outcome_type in outcomes:
    outcomes_df[outcome_type], maps[outcome_type], top10[outcome_type] = {}, {}, {}
    for pf in product_features:
          #compute the outcomes:
          outcomes_df[outcome_type][pf], maps[outcome_type][pf] = outcome_by_group_features(df, [pf, 'campaign_id'], outcome_type, min_sessions)

          #plot using heatmap
          sns.heatmap(maps[outcome_type][pf], cmap="RdBu_r", annot=False, fmt="0.0f")
          plt.title(f'{outcome_type} for campaign and {pf}')
          plt.show()

          #get the top10 pairs:
          top10[outcome_type][pf] = outcomes_df[outcome_type][pf].nlargest(n=10, columns=outcome_type)

          #print the top10:
          print(f'{outcome_type}, {pf}:')
          print(top10[outcome_type][pf])

  return outcomes_df, maps, top10

def plot_CR_vs_sessions (data, x_win = [(0, 100), (0, 1000)]):
  """
  Plot click-rate vs number of sessions for data segments

  Args:
    data - dataframe created by outcome_by_group_features()
    x_win - list of 2 tuples containing the windows (x_min, xmax) to view in each subplot

  Returns:
    pearson_r, p_value of click_rate(sessions_num)
  """

  fig, axes = plt.subplots(1,2)
  for j in range(2):
    sns.scatterplot(data = data, x = 'sessions_number', y = 'click_rate', ax=axes.flatten()[j]);
    axes.flatten()[j].set_xlim(x_win[j])
  fig.suptitle('click rate vs. session number')
  plt.tight_layout()
  plt.show()
  pearson_r, p_value = pearsonr(data.sessions_number, data.click_rate)
  #print(f"r={pearson_r} p = {p_value:.4f}")
  return pearson_r, p_value

#-------------------------------------------------------
#                       Part 3
#-------------------------------------------------------
def convert_to_date_time_df (df, DateTime):
  df = df.copy()
  df[DateTime] = pd.to_datetime(df[DateTime], dayfirst=True)
  #print(df[DateTime].isnull())
  df.dropna(subset=[DateTime,'is_click'], inplace=True)
  df.set_index(DateTime, inplace = True)
  df.sort_index(inplace=True)
  return df

#-------------------------------------------------------
#                       Part 4
#-------------------------------------------------------

def rate_of_N_clickers (original_df, N_clicks):
  """ get an original dataframe and produce a data frame with rates of unqiue users clicking from 1 to N times(inclusive)
  """
  df = original_df.dropna(subset=['user_id','campaign_id','is_click'])
  result = df.groupby(['campaign_id'])['user_id'].nunique().rename('exposure')
  clicks_by_users = df.groupby(['user_id'])['is_click'].sum()

  for N in range(N_clicks+1):
    clickers = clicks_by_users[clicks_by_users>N]
    clickers_ids = clickers.index
    df_clickers = df.loc[df['user_id'].isin(clickers_ids)]
    clickers_grouped = df_clickers.groupby(['campaign_id'])['user_id'].nunique().rename('clickers_num')
    campaign_clickers = pd.concat([result, clickers_grouped],axis=1)
    clickers_rate = campaign_clickers['clickers_num']/campaign_clickers['exposure']
    clickers_rate.name = f'{N}'
    result = pd.concat([result,clickers_rate],axis=1)

  result.index = result.index.astype(int).astype(str)
  return result