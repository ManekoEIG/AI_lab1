"""Метрики, матрица ошибок и кривые обучения для части 2."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
from common import OUT, CLASSES


def evaluate_and_save(tag, title, y_true, y_pred, hist, extra=None):
    acc = accuracy_score(y_true, y_pred)
    res = {
        "accuracy": acc,
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "per_class_precision": precision_score(y_true, y_pred, average=None).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, average=None).tolist(),
        "errors": int((y_true != y_pred).sum()),
        "history": hist,
    }
    if extra:
        res.update(extra)
    cm = confusion_matrix(y_true, y_pred)
    res["confusion_matrix"] = cm.tolist()

    print(f"\n=== {title}: test set ({len(y_true)} images) ===")
    print(f"Accuracy  = {acc*100:.2f}%")
    print(f"Precision = {res['precision_macro']*100:.2f}% (macro)")
    print(f"Recall    = {res['recall_macro']*100:.2f}% (macro)")
    print(f"F1        = {res['f1_macro']*100:.2f}% (macro)")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix:\n", cm)

    with open(os.path.join(OUT, f"{tag}_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)

    # Матрица ошибок
    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8.5)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"Матрица ошибок — {title} (accuracy {acc*100:.2f}%)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{tag}_confusion_matrix.png"), dpi=150)
    plt.close()

    # Кривые обучения
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(ep, hist["train_loss"], label="train")
    axes[0].plot(ep, hist["val_loss"], label="val")
    axes[0].set_title("Функция потерь"); axes[0].set_xlabel("эпоха"); axes[0].legend()
    axes[1].plot(ep, hist["train_acc"], label="train")
    axes[1].plot(ep, hist["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("эпоха"); axes[1].legend()
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{tag}_curves.png"), dpi=150)
    plt.close()
    return res
