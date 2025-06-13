# eval.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_fscore_support,
                             cohen_kappa_score, jaccard_score, accuracy_score)

# Global DataFrame to store evaluation metrics
crdf = pd.DataFrame()

# Categories or class labels (update if needed)
categories = ['meningioma', 'glioma', 'pituitary']

# Evaluate model predictions and display classification metrics
def evaluate_model(model, X_test, y_test, model_name):
    print("Generating predictions...")
    pred_Y = model.predict(X_test, batch_size=32, verbose=True)
    pred_Y_cat = np.argmax(pred_Y, axis=-1)
    test_Y_cat = np.argmax(y_test, axis=-1)

    print("Unique classes in test_Y_cat:", len(np.unique(test_Y_cat)))
    print("Unique classes in pred_Y_cat:", len(np.unique(pred_Y_cat)))

    # Confusion matrix
    cm = confusion_matrix(test_Y_cat, pred_Y_cat)
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap='Blues', fignum=1)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, f'{value}', ha='center', va='center',
                 color='white' if value > cm.max()/2 else 'black')
    plt.xticks(ticks=np.arange(len(categories)), labels=categories)
    plt.yticks(ticks=np.arange(len(categories)), labels=categories)
    plt.colorbar()
    plt.show()

    print(classification_report(test_Y_cat, pred_Y_cat, target_names=categories, digits=4))

    # Additional metric calculation
    precision_dict, recall_dict, f1_dict, specificity_dict, sensitivity_dict = calculate_metrics(test_Y_cat, pred_Y_cat, cm)

    # Summary table
    summary_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Micro': [precision_dict['micro'], recall_dict['micro'], f1_dict['micro']],
        'Macro': [precision_dict['macro'], recall_dict['macro'], f1_dict['macro']],
        'Weighted': [precision_dict['weighted'], recall_dict['weighted'], f1_dict['weighted']]
    })

    print("\nDifferences between averaging methods:")
    print("- Micro: Global computation over all instances.")
    print("- Macro: Average of metrics for each class.")
    print("- Weighted: Macro weighted by class support.")

    return summary_df

def calculate_metrics(y_true, y_pred, cm):
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    specificity_dict = {}
    sensitivity_dict = {}

    for avg in ['micro', 'macro', 'weighted']:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
        precision_dict[avg] = precision
        recall_dict[avg] = recall
        f1_dict[avg] = f1

    for label in np.unique(y_true):
        TP = cm[label, label]
        FN = cm[label, :].sum() - TP
        FP = cm[:, label].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        specificity_dict[label] = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivity_dict[label] = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision_dict, recall_dict, f1_dict, specificity_dict, sensitivity_dict

def visualise(y_true, y_pred, name, model_name):
    ConfusionM = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=name, digits=4))

    plt.figure(figsize=(6, 6))
    sns.heatmap(ConfusionM / ConfusionM.astype(float).sum(axis=0), annot=True,
                fmt='0.2f', cmap='Greys', xticklabels=name, yticklabels=name)
    plt.title('Confusion Matrix for ' + model_name)
    plt.savefig('CM_' + model_name + '.jpg')
    plt.show()

    global crdf
    report = classification_report(y_true, y_pred, target_names=name, output_dict=True)
    temp_df = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy_score(y_true, y_pred)],
        "Cohen's Kappa": [cohen_kappa_score(y_true, y_pred)],
        'Jaccard Similarity': [jaccard_score(y_true, y_pred, average='macro')]
    })

    for label in name:
        temp_df[f'{label}_Precision'] = report[label]['precision']
        temp_df[f'{label}_Recall'] = report[label]['recall']
        temp_df[f'{label}_F1-Score'] = report[label]['f1-score']

    crdf = pd.concat([crdf, temp_df], ignore_index=True)

def pred(model, X, Y, encoder):
    yhat = model.predict(X)
    yhat = encoder.inverse_transform(yhat)
    y_real = encoder.inverse_transform(Y)
    return y_real, yhat
