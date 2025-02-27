"""
Constants for data splitting configuration.

This module holds the default configuration for splitting the dataset into training,
validation, and test sets. It includes default split ratios, the test product, and the
test user group. It also defines whether to leave out the designated product or user group
from training and the default random state for reproducible splits.
"""

# Default ratios for training, validation, and test splits.
TRAIN_TEST_VAL_SPLIT = (0.6, 0.2, 0.2)

# The product that should be withheld from the training set.
PRODUCT_TO_LEAVE = 'A'
# Default behaviors for excluding data from training.
DEFAULT_LEAVE_ONE_PRODUCT_OUT = True      # By default, exclude sessions with the specified product.
LEAVE_ONE_PRODUCT_OUT = True

# Default random state for reproducible splits.
RANDOM_STATE = 42 