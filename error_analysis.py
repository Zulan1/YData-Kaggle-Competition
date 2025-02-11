import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from app.metrics import compute_score, supported_metrics
from predict import get_model, get_data, get_ohe, transform_categorical_columns
from app.helper_functions import split_dataset_Xy
from preprocess import feature_engineering, preprocess_towards_training
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
import constants as cons


def get_external_data (input_path, run_id) -> tuple:
    X = pd.read_csv(os.path.join(input_path, 'X_test_1st_raw.csv'))
    y = pd.read_csv(os.path.join(input_path, 'y_test_1st.csv'), header=None)
    return X, y

def preprocess_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=[col for col in df.columns if col not in cons.COLUMNS_TO_DROP])
    df, ohe = preprocess_towards_training(df)
    return df, ohe

def preprocess_features_for_prediction(ohe, model, df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(df.mode().iloc[0])
    df = feature_engineering(df)
    
    df = transform_categorical_columns(df, ohe)

    train_features = model.feature_names_in_
    print ('Model features:', train_features)
    print ('The following columns were seen during training but are not present in the external test-set:')
    for feature in train_features:
        if feature not in df.columns:
            print(f'- {feature}')
            df[feature] = 0
    print ('The following columns are present in the external test-set but were not seen during training:')
    for feature in df.columns:
        if feature not in train_features:
            print(f'- {feature}')
            df.drop(columns=feature, inplace=True)
    return df

def compute_metrics(y_true, y_pred, y_proba) -> dict:

    output = {}
    for metric in supported_metrics:
        try:
            score = compute_score(metric, y, y_pred, y_proba)
            output[metric] = score
        except ValueError as e:
            output[metric] = None
            continue
    cmat = confusion_matrix(y, y_pred)
    cmat_normalized = 100 * cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
    output['cmat'] = cmat
    output['cmat_normalized'] = cmat_normalized
    return output

def print_metrics(metrics):
    for metric, score in metrics.items():
        if metric == 'cmat_normalized':
            print('Normalized confusion matrix:')
            print(np.array2string(score, formatter={'float_kind':lambda x: f"{x:.1f}%"}))
        elif metric == 'cmat':
            print('Confusion matrix:')
            print(score)
        else:
            print(f"{metric}: {score:.4f}")


run_id = '1739223824'
input_path = './data/'
output_path = './data/'




print("Loading model")
model = get_model(input_path, run_id)


print(f"Model type: {model.__class__.__name__}")
print(f"Loading data from {input_path}")

X, y = get_external_data(input_path, run_id)
ohe = get_ohe(input_path, run_id)
X = preprocess_features_for_prediction(ohe, model, X)


print(f"Predicting {cons.TARGET_COLUMN} for {input_path} (model was trained on 60% of the data)")

y_pred = model.predict(X)
y_pred = pd.DataFrame(y_pred, index=X.index, columns=[cons.TARGET_COLUMN])
y_proba = model.predict_proba(X)[:, 1]
y_proba = pd.DataFrame(y_proba, index=X.index, columns=['proba'])


#print(f"Predicted {cons.TARGET_COLUMN} for {input_path}")

output_dir = os.path.join(output_path, f"err_analysis_{run_id}")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{cons.DEFAULT_PREDICTIONS_FILE}_partial.csv")
y_pred.to_csv(output_path, index=False)



# Compute metrics for partial training:

print(' Full training:')
#Full training:
#Load the full training dataset
df = pd.read_csv(os.path.join(input_path, 'train_dataset_full.csv'))


#Preprocess the data:
df, ohe = preprocess_features_for_training(df)
X_train, y_train = split_dataset_Xy(df)

print(X_train.shape)


#Fit the optimized model:
model2 = clone(model)
model2.set_params(verbose=1)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
model2.fit(X_train, y_train)

#Load the external test-set:
X, y = get_external_data(input_path, run_id)
X = preprocess_features_for_prediction(ohe, model2, X)


#Predict on the external test-set:
y_pred2 = model2.predict(X)
y_pred2 = pd.DataFrame(y_pred2, index=X.index, columns=[cons.TARGET_COLUMN])
y_proba2 = model2.predict_proba(X)[:, 1]
y_proba2 = pd.DataFrame(y_proba2, index=X.index, columns=['proba'])

output_path = os.path.join(output_dir, f"{cons.DEFAULT_PREDICTIONS_FILE}_full.csv")
y_pred2.to_csv(output_path, index=False)

print ('Partial training:')
metrics = compute_metrics(y, y_pred, y_proba)
print_metrics(metrics)
print('')
print ('Full training:')
#Compute metrics for full training:
metrics2 = compute_metrics(y, y_pred2, y_proba2)
print_metrics(metrics2)















