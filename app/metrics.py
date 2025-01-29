from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score

def compute_score(option, y_true, y_pred) -> float:
    """Compute the score of the model on the test/validation set.
    Args:
        option (str): The metric to use for evaluation.
        y_pred (np.array): The predicted target values.
        y_true (np.array): The true target values.
    Returns:
        float: The score of the model.
    """
    # Calculate the specified metric
    if option == 'f1':
        test_score = f1_score(y_true, y_pred, average='weighted')
    elif option == 'mcc':
        test_score = matthews_corrcoef(y_true, y_pred)
    elif option == 'bacc':
        test_score = balanced_accuracy_score(y_true, y_pred)
    elif option == 'aba':
        test_score = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    
    return test_score