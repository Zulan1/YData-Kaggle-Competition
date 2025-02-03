# Paths and filenames:
DATA_PATH = 'data/'
DEFAULT_RAW_TRAIN_FILE = 'train_dataset_full.csv'
DEFAULT_EXTERNAL_RAW_TEST_FILE = 'X_test_1st_raw.csv'
DEFAULT_ONE_HOT_ENCODER_FILE = 'ohe.pkl'

DEFAULT_TRAIN_SET_FILE = 'train.csv'
DEFAULT_VAL_SET_FILE = 'val.csv'
DEFAULT_TEST_SET_FILE = 'test.csv'
DEFAULT_HOLDOUT_FEATURES_FILE = 'holdout_features.csv'
DEFAULT_HOLDOUT_LABELS_FILE = 'holdout_labels.csv'

DEFAULT_PROCESSED_TEST_FILE = 'processed_test.csv'

DEFAULT_PREDICTIONS_FILE = 'predictions.csv'
DEFAULT_RESULTS_FILE = 'results.csv'

COLUMNS_TO_DROP = ['product_category_2', 'city_development_index']
DATETIME_COLUMN = 'DateTime'
TARGET_COLUMN = 'is_click'
INDEX_COLUMNS = ['session_id', 'user_id']

DEFAULT_IMPUTER_FILE = 'imputer.pkl'
DEFAULT_OHE_FILE = 'ohe.pkl'

COLUMNS_TO_IMPUTE = ['gender', 'age_level', 'user_group_id', 'product_category_1', 'product', 'campaign_id', 'webpage_id', 'var_1', 'DateTime']

#Column names groups:
DEMOGRAPHICS = ['gender', 'age_level', 'city_development_index', 'user_group_id']
CATEGORICAL = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'var_1', 'day_of_week']

#Split and Fold parameters:
RANDOM_STATE = 42
#TRAIN_TEST_SPLIT = 0.4  # Temp = 40% for validation + test
#VAL_TEST_SPLIT = 0.5    # Split Temp equally into validation and test

TRAIN_TEST_VAL_SPLIT = (0.2, 0.2, 0.6)
