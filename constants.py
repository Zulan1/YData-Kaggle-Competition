
# Paths and filenames:
DATA_PATH = 'data/'
DEFAULT_RAW_TRAIN_FILE = 'train_dataset_full.csv'
DEFAULT_EXTERNAL_RAW_TEST_FILE = 'X_test_1st_raw.csv'

DEFAULT_TRAIN_SET_FILE = 'train.csv'
DEFAULT_VAL_SET_FILE = 'val.csv'
DEFAULT_TEST_SET_FILE = 'test.csv'

COLUMNS_TO_DROP = ['product_category_2', 'city_development_index']
DATETIME_COLUMN = 'DateTime'
TARGET_COLUMN = 'is_click'

#Column names groups:
DEMOGRAPHICS = ['gender', 'age_level', 'city_development_index', 'user_group_id']
CATEGORIAL = ['day_of_week', 'product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'var_1']


#Split and Fold parameters:
RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.4  # Temp = 40% for validation + test
VAL_TEST_SPLIT = 0.5    # Split Temp equally into validation and test

