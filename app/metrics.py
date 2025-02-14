from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc, confusion_matrix, fbeta_score
    )

import numpy as np

supported_metrics = ['f1', 'mcc', 'bacc',
                      'aba', 'auc', 'auprc',
                        'precision', 'recall','f-0.5']

def compute_score(option, y_true, y_pred, y_proba=None, suppressError = True) -> float:
    """Compute the score of the model on the test/validation set.
    Args:
        option (str): The metric to use for evaluation.
        y_pred (np.array): The predicted target values.
        y_true (np.array): The true target values.
        y_proba (np.array, optional): The predicted probabilities (needed for AUC and AUPRC).
    Returns:
        float: The score of the model.
    """
    # Calculate the specified metric
    if option == 'f1':
        test_score = f1_score(y_true, y_pred)

    elif option == 'mcc':
        test_score = matthews_corrcoef(y_true, y_pred)

    elif option == 'bacc':
        test_score = balanced_accuracy_score(y_true, y_pred)

    elif option == 'aba':
        test_score = balanced_accuracy_score(y_true, y_pred, adjusted=True)

    elif option == 'auc':
        # AUC requires predicted probabilities, so make sure it's provided
        if y_proba is None:
            if suppressError:
                return np.nan
            else:
                raise ValueError("AUC requires predicted probabilities (y_proba) to be passed.")
        test_score = roc_auc_score(y_true, y_proba)

    elif option == 'auprc':
        # AUPRC requires predicted probabilities, so make sure it's provided
        if y_proba is None:
            if suppressError:
                return np.nan
            else:
                raise ValueError("AUPRC requires predicted probabilities (y_proba) to be passed.")
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        test_score = auc(recall, precision)
    
    elif option == 'precision':
        test_score = precision_score(y_true, y_pred, zero_division=0)
    
    elif option == 'recall':
        test_score = recall_score(y_true, y_pred, zero_division=0)

    elif option.startswith('f-'):
        beta = float(option.split('-')[1])
        test_score = fbeta_score(y_true, y_pred, beta=beta)

    else:
        raise ValueError(f"Invalid scoring method: {option}")
    
    return test_score

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
    cmat_normalized = 100 * cmat.astype('float') / cmat.sum()
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

