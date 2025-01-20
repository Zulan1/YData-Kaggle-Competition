import argparse
import cons
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', default=cons.DATA_PATH, type=str, help='Path to input CSV file')
    parser.add_argument('--train', type=str, default=cons.DEFAULT_PROCESSED_TRAIN_FILE, help='Training data')
    parser.add_argument('--test', type=str, default=cons.DEFUALT_PROCESSED_TEST_FILE, help='Test data')
    parser.add_argument('--verbose', type=bool, default=True, help='Print additional information')
    #parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    #parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    #Load processed data:
    train_data = pd.read_csv(args.csv_path + '/' + args.train)
    test_data = pd.read_csv(args.csv_path + '/' + args.test)

    # Split the data into features and target:
    X_train, X_test = [a.drop(columns=['is_click']) for a in [train_data, test_data]]
    y_train, y_test = [a['is_click'] for a in [train_data, test_data]]

    # Model
    if args.verbose:
        print('Training model: stratified DummyClassifier')
    model = DummyClassifier(strategy='stratified') # DummyClassifier supports NaN by default
    model.fit(X_train, y_train)

    # Model
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 Score: {f1:.3f}')


if __name__ == '__main__':
    main()
