# src/compute_metrics.py

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def compute_metrics(predictions, labels, threshold=0.5, ctn=30):
    """
    Computes True Positives (TP), False Negatives (FN), and False Positives (FP)
    based on custom definitions.

    Args:
        predictions (np.ndarray): Array of predictions with shape (n_samples, n_classes).
        labels (np.ndarray): Array of ground truth labels with shape (n_samples, n_classes).
        threshold (float): Threshold to binarize predictions.
        ctn (int): Context window size for metric computation.

    Returns:
        tuple: (TP, FN, FP)
    """
    n, l = labels.shape

    # Binarize predictions based on threshold
    pred_binary = (predictions >= threshold).astype(int)

    # Initialize counts
    TP = 0
    FP = 0
    FN = 0

    for i in range(n):
        for j in range(l):
            if labels[i, j] == 1:
                # Check if any prediction within the context window meets the condition
                start = max(j - ctn, 0)
                end = min(j + ctn + 1, l)
                if pred_binary[i, start:end].any():
                    TP += 1
                else:
                    FN += 1
            else:
                # For FP, check if prediction is positive where label is negative
                if pred_binary[i, j] == 1:
                    FP += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_socre = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_socre:.4f}")
    report_str = {"Precision": precision, "Recall": recall, "F1 Score": f1_socre}
    return report_str
