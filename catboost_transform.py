import config as conf
import pandas as pd

def catboost_transform(df: pd.DataFrame):
    df = df.copy()
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype("string[python]")
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(int)
    print("df types after cb transform", df.dtypes)
    return df