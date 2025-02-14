import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from app.metrics import compute_score, supported_metrics
#from predict import get_model
from app.helper_functions import split_dataset_Xy
#from preprocess import feature_engineering, preprocess_towards_training
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import constants as cons
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import catboost


# def load_results(path, run_id):
#     path1 = os.path.join(path, f'err_analysis_{run_id}')
#     results_path = os.path.join(path1, 'results.pkl')
#     return pickle.load(open(results_path, 'rb'))

# def get_external_data (input_path, run_id) -> tuple:
#     X = pd.read_csv(os.path.join(input_path, 'X_test_1st_raw.csv'))
#     y = pd.read_csv(os.path.join(input_path, 'y_test_1st.csv'), header=None)
#     return X, y

# def preprocess_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.drop_duplicates()
#     df = df.dropna(subset=[col for col in df.columns if col not in cons.COLUMNS_TO_DROP])
#     df, ohe = preprocess_towards_training(df)
#     return df, ohe

# def preprocess_features_for_prediction(ohe, model, df: pd.DataFrame) -> pd.DataFrame:
#     df = df.fillna(df.mode().iloc[0])
#     df = feature_engineering(df)
    
#     df = transform_categorical_columns(df, ohe)

#     train_features = model.feature_names_in_
#     print ('Model features:', train_features)
#     print ('The following columns were seen during training but are not present in the external test-set:')
#     for feature in train_features:
#         if feature not in df.columns:
#             print(f'- {feature}')
#             df[feature] = 0
#     print ('The following columns are present in the external test-set but were not seen during training:')
#     for feature in df.columns:
#         if feature not in train_features:
#             print(f'- {feature}')
#             df.drop(columns=feature, inplace=True)
#     return df

def compute_metrics(y_true, y_pred, y_proba = None) -> dict:

    output = {}
    for metric in supported_metrics:
        try:
            score = compute_score(metric, y_true, y_pred, y_proba)
            output[metric] = score
        except ValueError as e:
            output[metric] = None
            continue
    cmat = confusion_matrix(y_true, y_pred)
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

def train_predict_full_model(input_path, run_id, optimized_model, output_dir, drop_features = None):
    # Full training:
    # Load the full training dataset
    df = pd.read_csv(os.path.join(input_path, 'train_dataset_full.csv'))

    # Preprocess the data:
    df, ohe = preprocess_features_for_training(df)
    X_train, y_train = split_dataset_Xy(df)
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    # Drop columns if needed:
    if drop_features:
        X_train.drop(columns=drop_features, inplace=True, errors = 'ignore')
        
    

    # Fit the optimized model:
    model = clone(optimized_model)
    model.set_params(verbose=1)
    model.fit(X_train, y_train)

    # Load the external test-set:
    X, y = get_external_data(input_path, run_id)
    X = preprocess_features_for_prediction(ohe, model, X)

    # Drop columns if needed:
    if drop_features:
        X.drop(columns=drop_features, inplace=True, errors='ignore')

    # Predict on the external test-set:
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, index=X.index, columns=[cons.TARGET_COLUMN])
    y_proba = model.predict_proba(X)[:, 1]
    y_proba = pd.DataFrame(y_proba, index=X.index, columns=['proba'])
    # Join y_pred with y_proba
    results = pd.concat([y_pred, y_proba], axis=1)

    # Save results:
    output_path = os.path.join(output_dir, f'err_analysis_{run_id}')
    results.to_csv(os.path.join(output_path, 'predictions_proba.csv'), index=False)
    model_output_path = os.path.join(output_path, 'fully_trained_model.pkl')
    results_path = os.path.join(output_path, 'results.pkl')

    with open(model_output_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    results_dict = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model_path': model_output_path,
        'df': df,
        'ohe': ohe,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X,
        'y_test': y
    }
    with open(results_path, 'wb') as results_file:
        pickle.dump(results_dict, results_file)

    return results_dict

def perm_feature_importance(model, X_test, y_test, output_path, run_id, n_repeats = 10, random_state=None):
    
    # Compute permutation feature importance
    perm_importance = permutation_importance(model, X_test, y_test, n_jobs = -1, scoring= 'f1', n_repeats=10, random_state=random_state)

    # Create DataFrame
    perm_importance_df = pd.DataFrame(np.abs(perm_importance.importances_mean), index=X_test.columns, columns=['Importance']).reset_index().rename(columns={'index':'feature'})

    # Save permutation importance to CSV
    output_dir = os.path.join(output_path, f"err_analysis_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    perm_importance_df.to_csv(os.path.join(output_dir, 'perm_importance.csv'), index=False)

    print("Permutation importance computed and saved.")

    return perm_importance_df

def shap_feature_importance(model, X_test, y_test, output_path, run_id, subset_frac = 0.01):

    # Sample data for SHAP analysis
    # X_subset = X_test.sample(frac=0.01, random_state=0)
    indices = np.arange(X_test.shape[0])
    subset = np.random.choice(indices, size=int(subset_frac * len(indices)), replace=False)
    X_subset = X_test.iloc[subset]
    y_subset = y_test.iloc[subset]
    explainer = shap.TreeExplainer(model)
    print ('Computing SHAP values (this can take some time...)')
    shap_values = explainer.shap_values(X_subset)
   
    y = y_subset.to_numpy().reshape(-1, 1)
    SVprob = shap_values[:,:,1] + explainer.expected_value[1]
    nlogloss_SVprob = -((y * np.log(SVprob) + (1 - y) * np.log(1 - SVprob)))
    # for false positive score, take only the y=0 instances
    y_FP = y.copy()[y == 0].reshape(-1,1)
    SVprob_FP = SVprob.copy()
    SVprob_FP = SVprob_FP[y.flatten() == 0, :]
    nlogloss_SVprob_FP = -((y_FP * np.log(SVprob_FP) + (1 - y_FP) * np.log(1 - SVprob_FP)))
    SV = np.mean(nlogloss_SVprob, axis=0)
    SV_FP = np.mean(nlogloss_SVprob_FP, axis=0)

    # Create DataFrame
    shap_values_df = pd.DataFrame({
        'feature': X_subset.columns,
        'SV': SV,
        'SV_FP': SV_FP,
        'SVc' : SV - SV.mean(),
        'SV_FPc' : SV_FP - SV_FP.mean()
    })

    # Save SHAP values to CSV
    output_dir = os.path.join(output_path, f"err_analysis_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    shap_values_df.to_csv(os.path.join(output_dir, 'shap_log_loss.csv'), index=False)
    SVprob_df = pd.DataFrame(SVprob, columns = X_subset.columns)
    SVprob_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)

    print("SHAP values computed and saved.")
    return shap_values_df, SVprob_df


# run_id = '1739223824'
input_path = './data/'
output_path = './data/'

# print("Loading model")
# model = get_model(input_path, run_id)

# print(f"Model type: {model.__class__.__name__}")
# print(f"Loading data from {input_path}")

# output_dir = os.path.join(output_path, f"err_analysis_{run_id}")
# os.makedirs(output_dir, exist_ok=True)

# print ('Full training:')
# #results = train_predict_full_model(input_path, run_id, model, output_path)
# results = load_results(output_path, run_id)

# # Compute metrics for full training:
# metrics2 = compute_metrics(results['y_test'], results['y_pred'], results['y_proba'])
# print_metrics(metrics2)

# # Compute metrics for new nodel
# new_pred_path = os.path.join(input_path, 'new')
# y_test = pd.read_csv('./data/y_test_1st.csv', header=None)

#Load improved model
print("Loading improved model and predicting:")
model_imp = pickle.load(open('./data/new/full_model.pkl', 'rb'))
X_test_imp = pd.read_csv('./data/new/X_test.csv')
y_test_imp = pd.read_csv('./data/new/y_test.csv', header=None)

y_pred_imp = model_imp.predict(X_test_imp)
y_proba_imp = model_imp.predict_proba(X_test_imp)[:, 1]
metrics_imp = compute_metrics(y_test_imp, y_pred_imp, y_proba_imp)
print ('Metrics for improved model:')
print_metrics(metrics_imp)



# # print ('SHAP feature importance:')
# # # SHAP feature importance using a subset of the data
# # shap_feature_importance(results['model'], results['X_test'], output_path, run_id)


# # Permutation feature importance
# print ('Permutation feature importance:')
# recompute = False
# if recompute:
#     perm_importance = perm_feature_importance(results['model'], results['X_test'], results['y_test'], output_path, run_id)
# else:
#     perm_importance = pd.read_csv(os.path.join(output_dir, 'perm_importance.csv'))


# dbg = 0

# # Plot permutation feature importance as horizontal bar plot
# plt.figure(figsize=(10, 10))
# sns.barplot(data=perm_importance, y='feature', x='Importance', orient='h')
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.title('Permutation Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'perm_importance.png'))
# # plt.show()

# recompute = True
# if recompute:
#     shap_importance, SVprob = shap_feature_importance(results['model'], results['X_test'], results['y_test'], output_path, run_id, subset_frac = 1.0)
# else:
#     shap_importance = pd.read_csv(os.path.join(output_dir, 'shap_log_loss.csvs.csv'))
#     SVprob = pd.read_csv(os.path.join(output_dir, 'shap_values.csv'))

# plt.barh(shap_importance['feature'], shap_importance['SVc'], label = 'SV')
# plt.barh(shap_importance['feature'], shap_importance['SV_FPc'], label = 'FP SV')
# plt.yticks(fontsize=8)
# plt.xlabel('SHAP value')
# plt.ylabel('Feature')
# plt.title('SHAP values')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'shap_values.png'))
# # plt.show()


# diff_shap = shap_importance['SV'] - shap_importance['SV'].mean()
# # Find features with negative SHAP values
# extreme_shap_features = shap_importance[diff_shap > 0]

# # Sort the features by their SHAP values
# sorted_extreme_shap_features = extreme_shap_features.sort_values(by='SV')

# print("Features with positive SHAP values:")
# print(sorted_extreme_shap_features)

# print("Training the model without the features with negative SHAP values:")
# #Take all of these feautes and train the model without them:
# results2 = train_predict_full_model(input_path, run_id, model, output_path, drop_features=sorted_extreme_shap_features['feature'].to_list())

# metrics4 = compute_metrics(results2['y_test'], results2['y_pred'], results2['y_proba'])
# print('Model trained without features with negative SHAP values:')
# print_metrics(metrics4)