from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class MetricBundle:
    overall: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    confusion: List[List[int]]
    confidence_by_true: Dict[str, float]


def classification_metrics(y_true, y_pred, prob_fact, positive_label='Fact', negative_label='Opinion') -> MetricBundle:
    labels = [negative_label, positive_label]
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_f1 = np.mean(f1)

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_true_binary = np.array([label_to_idx[label] for label in y_true])
    # Probability of positive label (Fact) for Brier score
    prob = np.clip(np.array(prob_fact), 0.0, 1.0)
    brier = brier_score_loss(y_true_binary, prob)

    conf = confusion_matrix(y_true, y_pred, labels=labels)

    confidence_by_true = {}
    for label in labels:
        mask = np.array(y_true) == label
        if mask.any():
            confidence_by_true[label] = float(np.mean(prob[mask]))

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx]),
        }

    overall = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'brier': float(brier),
        'n': int(len(y_true)),
    }

    return MetricBundle(
        overall=overall,
        per_class=per_class,
        confusion=conf.tolist(),
        confidence_by_true=confidence_by_true,
    )
