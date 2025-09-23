import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict(X_test)[:, 0]
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        'report': report,
        'auc': auc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, labels, out_path=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
    plt.close()
