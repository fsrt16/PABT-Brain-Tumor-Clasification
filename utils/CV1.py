from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, jaccard_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global DataFrame to store results
crdf = pd.DataFrame()

def multiclass_roc_auc_score(y_test, y_pred, fold):
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    y_pred = lb.transform(y_pred)

    plt.subplot(3, 3, fold)
    plt.title(f'ROC Curve - Fold {fold}')
    for idx, label in enumerate(lb.classes_):
        fpr, tpr, _ = roc_curve(y_test[:, idx], y_pred[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC: {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

def visualise(y_true, y_pred, fold):
    ConfusionM = confusion_matrix(y_true, y_pred)

    plt.subplot(3, 3, fold)
    plt.title(f'Confusion Matrix - Fold {fold}')
    sns.heatmap(ConfusionM, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))

    accuracy = accuracy_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    jaccard_sim = jaccard_score(y_true, y_pred, average='macro')

    global crdf
    temp_df = pd.DataFrame({
        'Fold': [fold],
        'Accuracy': [accuracy],
        'Cohens Kappa': [cohen_kappa],
        'Jaccard Similarity': [jaccard_sim]
    })
    crdf = pd.concat([crdf, temp_df], ignore_index=True)

def map_class_labels(Y, class_labels):
    return [class_labels[e] for e in Y.flatten()]

def cross_validate(X, Y, model, enc, class_labels):
    skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)

    plt.figure(figsize=(15, 12))
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        y_pred = model.predict(X_test)
        y_pred = enc.inverse_transform(y_pred)
        visualise(Y_test, y_pred, fold)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 12))
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        y_pred = model.predict(X_test)
        y_pred = enc.inverse_transform(y_pred)
        multiclass_roc_auc_score(Y_test, y_pred, fold)
    plt.tight_layout()
    plt.show()

    return crdf

# Example usage:
# class_labels = {1: 'meningioma', 2: 'glioma', 3: 'pituitary'}
# dec_Y = enc.inverse_transform(Y)
# cross_validate(X, dec_Y, mammo_model, enc, class_labels)
