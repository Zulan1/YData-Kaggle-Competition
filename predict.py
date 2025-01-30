from app.argparser import get_predict_args
import pandas as pd
import pickle
import constants as cons
import os
from sklearn.impute import SimpleImputer

from preprocess import feature_engineering

def get_model(model_path):
    return pickle.load(open(model_path, 'rb'))

def get_data(input_data):
    return pd.read_csv(input_data)

def get_predictions(model, df):
    return model.predict(df)

def get_imputer(imputer_path):
    return pickle.load(open(imputer_path, 'rb'))

def transform_categorical_columns(df: pd.DataFrame, ohe_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Transform categorical columns using a pre-fitted OneHotEncoder.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        ohe_path (str): Path to the saved OneHotEncoder
        verbose (bool): Whether to print verbose output
        
    Returns:
        pd.DataFrame: DataFrame with transformed categorical columns
    """
    # Load the one-hot encoder
    with open(ohe_path, 'rb') as f:
        ohe = pickle.load(f)
    
    if verbose:
        print(f"Loaded OneHotEncoder from {ohe_path}")
    
    # Transform categorical columns
    encoded_cats = ohe.transform(df[cons.CATEGORICAL])
    feature_names = ohe.get_feature_names_out(cons.CATEGORICAL)
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    
    # Drop original categorical columns and join encoded ones
    df = df.drop(columns=cons.CATEGORICAL)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df

def main():
    args = get_predict_args()

    model = get_model(args.model_path)
    if args.verbose:
        print(f"Loading model from {args.model_path}")
    

    df = get_data(args.input_data)
    if args.verbose:
        print(f"Loading data from {args.input_data}")
        
    df = df.fillna(df.mode().iloc[0])
    df.drop(columns=cons.INDEX_COLUMNS, inplace=True)
    df = feature_engineering(df, args.verbose)
    df = transform_categorical_columns(df, args.ohe_path, args.verbose)

    predictions = get_predictions(df)
    if args.verbose:
        print(f"Predicted {cons.TARGET} for {args.input_data}")
    # Add predictions to the DataFrame
    df[cons.TARGET] = predictions

    # Save predictions to CSV
    predictions_path = os.path.join(os.path.dirname(args.predictions_path), 'predictions.csv')
    df.to_csv(predictions_path, index=False)
    
    if args.verbose:
        print(f"Predictions saved to {predictions_path}")

if __name__ == '__main__':
    main()
