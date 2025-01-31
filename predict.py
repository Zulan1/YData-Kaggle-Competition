from app.argparser import get_predict_args
import pandas as pd
import pickle
import constants as cons
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from preprocess import feature_engineering

def get_model(input_path: str, run_id: str):
    model_path = os.path.join(input_path, f"train_{run_id}/model.pkl")
    return pickle.load(open(model_path, 'rb'))

def get_data(input_path: str, run_id: str) -> pd.DataFrame:
    data_path = os.path.join(input_path, f"preprocess_{run_id}/{cons.DEFAULT_PROCESSED_TEST_FILE}")
    return pd.read_csv(data_path)

def transform_categorical_columns(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """Transform categorical columns using a pre-fitted OneHotEncoder.
    
    Args:
        df: Input DataFrame
        ohe: Pre-fitted OneHotEncoder
        
    Returns:
        DataFrame with transformed categorical columns
    """
    encoded_cats = ohe.transform(df[cons.CATEGORICAL])
    feature_names = ohe.get_feature_names_out(cons.CATEGORICAL)
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    df = df.drop(columns=cons.CATEGORICAL)
    return pd.concat([df, encoded_df], axis=1)

def get_ohe(input_path: str, run_id: str) -> OneHotEncoder:
    ohe_path = os.path.join(input_path, f"preprocess_{run_id}/ohe.pkl")
    return pickle.load(open(ohe_path, 'rb'))

def main():
    args = get_predict_args()
    run_id = args.run_id

    if args.verbose:
        print("Loading model")
    model = get_model(args.input_path, run_id)
    
    if args.verbose:
        print(f"Model type: {model.__class__.__name__}")
        print(f"Loading data from {args.input_path}")
    
    df = get_data(args.input_path, run_id)
    df = df.fillna(df.mode().iloc[0])
    df = feature_engineering(df)
    
    ohe = get_ohe(args.input_path, run_id)
    df = transform_categorical_columns(df, ohe)

    if args.verbose:
        print(f"Predicting {cons.TARGET_COLUMN} for {args.input_path}")
    
    print(df.columns)

    predictions = model.predict(df)
    predictions = pd.DataFrame(predictions, index=df.index, columns=[cons.TARGET_COLUMN])
    
    if args.verbose:
        print(f"Predicted {cons.TARGET_COLUMN} for {args.input_path}")

    output_dir = os.path.join(args.output_path, f"predict_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cons.DEFAULT_PREDICTIONS_FILE)
    predictions.to_csv(output_path, index=False)

    if args.verbose:
        print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()
