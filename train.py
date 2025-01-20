import argparse
import constants as cons
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

import sys
import os
# Add the 'app/' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), './app/'))

from helper_functions import load_training_data, split_dataset_Xy

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--csv-path', default=cons.DATA_PATH, type=str, help='Path to input CSV file')
    # parser.add_argument('--train', type=str, default=cons.DEFAULT_PROCESSED_TRAIN_FILE, help='Training data')
    # parser.add_argument('--test', type=str, default=cons.DEFUALT_PROCESSED_TEST_FILE, help='Test data')
    parser.add_argument('--verbose', type=bool, default=True, help='Print additional information')
    #parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    #parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    #Load processed data:
    folds, train_set, test_set = load_training_data()

    #Hyperparameters for the model:
    param_grid = {
        'strategy': ['stratified', 'most_frequent', 'uniform', 'constant'],
        'constant': [0]  # Only used if strategy='constant'
       
    }

    

    # Initialize a list of dummy classifiers:
    dummy_clfs = [DummyClassifier(strategy=strategy, constant=0) for strategy in param_grid['strategy']]

    # Iterate over pre-computed folds
    grid_scores = []
    for clf in dummy_clfs:
        fold_scores = []
        for (train_fold, val_fold) in folds:
            #Fit dummy classifier to the training fold
            X_train_fold, y_train_fold = split_dataset_Xy(train_fold)
            
            clf.fit(X_train_fold, y_train_fold)

            # Make predictions on the validation fold
            X_val, y_val = split_dataset_Xy(val_fold)
            y_val_pred = clf.predict(X_val)

            # Calculate F1 score
            f1 = f1_score(y_val, y_val_pred, average='weighted')
            fold_scores.append(f1)

        fold_scores = np.array(fold_scores)
        grid_scores.append(np.mean(fold_scores))  
    grid_scores = np.array(grid_scores)    

    # Print the best parameters and best score
    if args.verbose:
        print(f'Best F1: {np.max(grid_scores)}')
        print(f'Best Strategy: {param_grid["strategy"][np.argmax(grid_scores)]}')

    # Train the best model on the entire training set and evaluate on the test set
    best_model = dummy_clfs[np.argmax(grid_scores)]

    # Split the entire training set into features and target
    X_train, y_train = split_dataset_Xy(train_set)

    # Split the test set into features and target
    X_test, y_test = split_dataset_Xy(test_set)

    # Fit the best model on the entire training set
    best_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_test_pred = best_model.predict(X_test)

    test_f1_score = f1_score(y_test, y_test_pred, average='weighted')
    if args.verbose:
        print(f'Test F1 Score: {test_f1_score}')    
            
if __name__ == '__main__':
    main()
