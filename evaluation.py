import numpy as np
import torch


def accuracy(outputs, labels):
    assert len(outputs) == len(labels)

    if len(labels) == 0:
        return 0.0
    
    if isinstance(outputs, np.ndarray):
        preds = outputs.argmax(axis=1)
        return (preds == labels).mean()
    elif isinstance(outputs, torch.Tensor):
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
        return (preds == labels).float().mean().item()
    else:
        raise TypeError("outputs must be np.ndarray or torch.Tensor")


def precision_and_recall(outputs, labels):
    if labels is None or len(labels) == 0:
        return

    n = len(outputs)
    tp, fp, fn = 0, 0, 0
    for i in range(n):
        if labels[i] == 1 and outputs[i] == 1:
            tp += 1
        elif labels[i] == 0 and outputs[i] == 1:
            fp += 1
        elif labels[i] == 1 and outputs[i] == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def f1_score(outputs, labels):
    if labels is None or len(labels) == 0:
        return 0.0
    
    precision, recall = precision_and_recall(outputs, labels)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1