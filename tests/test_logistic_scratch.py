"""
Tests de la logistica desde cero - convergencia y copias defensivas.
"""

import numpy as np
from src.models.logistic_scratch import LogisticRegressionScratch, _sigmoid, _log_loss
from src.data.dataset import preprocess


def test_sigmoid_and_loss_stable():
    assert abs(_sigmoid(np.array([0]))[0] - 0.5) < 1e-9
    # extremos no deben dar inf/nan
    assert np.all(np.isfinite(_sigmoid(np.array([-1000, 1000]))))
    assert _log_loss(np.array([0, 1]), np.array([0.1, 0.9])) < 0.5


def test_defensive_copies():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    X_orig, y_orig = X.copy(), y.copy()
    m = LogisticRegressionScratch(lr=0.1, n_iter=5)
    m.fit(X, y)
    assert np.array_equal(X, X_orig), "fit muto X"
    assert np.array_equal(y, y_orig), "fit muto y"
    # predict tampoco debe mutar
    X2 = X.copy()
    m.predict(X2)
    assert np.array_equal(X2, X)


def test_convergence_loss_decreases():
    out = preprocess(test_size=0.2, random_state=42)
    X_train, y_train = out["X_train"].values, out["y_train"].values
    m = LogisticRegressionScratch(lr=0.5, n_iter=500, tol=1e-9)
    m.fit(X_train, y_train)
    # loss debe bajar monotona en general (permitimos ruido pequeno)
    assert m.loss_history_[0] > m.loss_history_[-1]
    assert m.loss_history_[-1] < 0.2
    assert len(m.loss_history_) <= 500


def test_predict_shapes_and_proba():
    out = preprocess(test_size=0.2, random_state=42)
    X_train, X_test = out["X_train"].values, out["X_test"].values
    y_train = out["y_train"].values
    m = LogisticRegressionScratch(lr=0.5, n_iter=200)
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)
    pred = m.predict(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(pred)).issubset({0, 1})
    # threshold custom
    pred_high = m.predict(X_test, threshold=0.9)
    assert pred_high.sum() <= pred.sum()


def test_class_weight_increases_recall():
    out = preprocess(test_size=0.2, random_state=42)
    X_train, X_test = out["X_train"].values, out["X_test"].values
    y_train, y_test = out["y_train"].values, out["y_test"].values

    plain = LogisticRegressionScratch(lr=0.5, n_iter=500)
    plain.fit(X_train, y_train)
    bal = LogisticRegressionScratch(lr=0.5, n_iter=500, class_weight="balanced")
    bal.fit(X_train, y_train)

    from sklearn.metrics import recall_score

    rec_plain = recall_score(y_test, plain.predict(X_test))
    rec_bal = recall_score(y_test, bal.predict(X_test))
    assert rec_bal > rec_plain, "balanced debe mejorar recall con 3% de fallas"
