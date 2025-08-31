import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(model_name: str, y_true: List[int], y_pred: List[int], labels: List[int] = None,
                          output_root: str = "results/images/confusion") -> str:
    """Save confusion matrix plot and return its path.

    Parameters
    ----------
    model_name : str
        Name of the model (used for file naming)
    y_true : List[int]
        Ground-truth labels
    y_pred : List[int]
        Predicted labels
    labels : List[int], optional
        Explicit label order; if None, will infer from `y_true` ∪ `y_pred` sorted asc
    output_root : str, default "results/images/confusion"
        Directory to drop images in.
    """
    if labels is None:
        labels = sorted({*set(y_true), *set(y_pred)})

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            ylabel="True label", xlabel="Predicted label",
            title=f"Confusion Matrix – {model_name}")

    # Rotate tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, f"{model_name}_confusion.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path 