
# Paths and filenames:
DATA_PATH = 'data/'
DEFAULT_RAW_TRAIN_FILE = 'train_dataset_raw.csv'
DEFAULT_EXTERNAL_RAW_TEST_FILE = 'X_test_1st_raw.csv'
DEFAULT_TRAIN_FOLD_FILE = 'train_fold_'
DEFAULT_VAL_FOLD_FILE = 'val_fold_'
DEFAULT_TRAIN_SET_FILE = 'train_set.csv'
DEFAULT_TEST_SET_FILE = 'test_set.csv'

#Column names groups:
DEMOGRAPHICS = ['gender', 'age_level', 'city_development_index', 'user_group_id']
CATEGORIAL = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'var_1']

#Split and Fold parameters:
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_FOLDS = 5