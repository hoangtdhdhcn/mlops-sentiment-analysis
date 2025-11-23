# evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Evaluates the model's performance using various metrics.
    
    Args:
    - y_true: The true labels of the test set.
    - y_pred: The predicted labels from the model.
    
    Returns:
    - metrics: A dictionary containing the evaluation metrics.
    """
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision (for each class and macro average)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', labels=np.unique(y_pred))
    
    # Recall (for each class and macro average)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', labels=np.unique(y_pred))
    
    # F1 Score (for each class and macro average)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_pred))
    
    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics
