"""
Tests de metricas - nunca accuracy sola con 3% de fallas.
"""

import numpy as np
from src.evaluation.metrics import compute_metrics


def test_compute_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])

    m = compute_metrics(y_true, y_pred, y_proba)
    # con 50% de falla, precision y recall conocidos
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"] <= 1
    assert 0 <= m["f1"] <= 1
    assert 0 <= m["roc_auc"] <= 1
    assert m["confusion"] == [[1, 1], [1, 1]]


def test_metrics_on_imbalanced():
    # simula el caso real: 96% no falla, 3% falla
    y_true = np.array([0] * 96 + [1] * 4)
    y_pred = np.array([0] * 96 + [1] * 4)  # perfecto
    m = compute_metrics(y_true, y_pred, np.array([0.1]*96 + [0.9]*4))
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0

    # tonto que siempre dice 0 - accuracy 96% pero recall 0
    y_pred_zero = np.zeros_like(y_true)
    m2 = compute_metrics(y_true, y_pred_zero, np.zeros_like(y_true, dtype=float))
    assert m2["recall"] == 0.0
    assert m2["precision"] == 0.0  # zero_division=0


def test_defensive_copy_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_true_c, y_pred_c = y_true.copy(), y_pred.copy()
    compute_metrics(y_true, y_pred)
    assert np.array_equal(y_true, y_true_c)
    assert np.array_equal(y_pred, y_pred_c)
