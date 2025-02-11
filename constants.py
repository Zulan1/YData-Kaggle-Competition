# Paths and filenames:
DATA_PATH = 'data/'
DEFAULT_INTERNAL_DATA_FILE = 'train_dataset_full.csv'
DEFAULT_EXTERNAL_RAW_TEST_FILE = 'X_test_1st_raw.csv'

DEFAULT_TRAIN_SET_FILE = 'train.csv'
DEFAULT_VAL_SET_FILE = 'val.csv'
DEFAULT_TRAIN_DTYPES_FILE = 'train_dtypes.pkl'
DEFAULT_VAL_DTYPES_FILE = 'val_dtypes.pkl'
DEFAULT_TEST_FEATURES_FILE = 'test_features.csv'
DEFAULT_TEST_DTYPES_FILE = 'test_dtypes.pkl'
DEFAULT_EXTERNAL_TEST_FEATURES_FILE = 'external_test_features.csv'
DEFAULT_EXTERNAL_TEST_DTYPES_FILE = 'external_test_dtypes.pkl'
DEFAULT_TEST_LABELS_FILE = 'test_labels.csv'

DEFAULT_PREDICTIONS_FILE = 'predictions.csv'
DEFAULT_PREDICTED_PROBABILITIES_FILE = 'predicted_probabilities.csv'

DEFAULT_RESULTS_FILE = 'results.csv'

DEFAULT_TRANSFORMER_FILE = 'transformer.pkl'
DEFAULT_MODEL_FILE = 'model.pkl'

COLUMNS_TO_DROP = ['product_category_2', 'city_development_index']
DATETIME_COLUMN = 'DateTime'
TARGET_COLUMN = 'is_click'
INDEX_COLUMNS = ['session_id', 'user_id']

COLUMNS_TO_OHE = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'var_1']

COLUMNS_TO_IMPUTE = ['gender', 'age_level', 'user_group_id', 'product_category_1', 'product', 'campaign_id', 'webpage_id', 'var_1', 'DateTime']

#Column names groups:
DEMOGRAPHICS = ['gender', 'age_level', 'city_development_index', 'user_group_id']
CATEGORICAL = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'var_1', 'day_of_week']
