import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, jaccard_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification_metrics(y_true, y_pred, average='macro'):
    """
    Compute and return multiple classification metrics.
    :param y_true: True class labels
    :param y_pred: Predicted class labels
    :param average: Averaging method for multi-class metrics
    :return: Dictionary of evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average),
        'Recall': recall_score(y_true, y_pred, average=average),
        'F1-Score': f1_score(y_true, y_pred, average=average),
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred),
        'Jaccard Index': jaccard_score(y_true, y_pred, average=average)
    }
    return metrics

def print_classification_report(metrics_dict):
    """
    Print a formatted report of evaluation metrics.
    :param metrics_dict: Dictionary of evaluation metrics
    """
    print("\nClassification Report:")
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(8, 6), cmap='Blues'):
    """
    Plot a confusion matrix heatmap.
    :param y_true: True class labels
    :param y_pred: Predicted class labels
    :param class_names: List of class names
    :param figsize: Size of the figure
    :param cmap: Color map for the heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_auc(y_true, y_score, n_classes):
    """
    Plot ROC AUC curves for multiclass classification.
    :param y_true: True class labels (one-hot encoded)
    :param y_score: Predicted probability scores
    :param n_classes: Number of output classes
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_test_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
