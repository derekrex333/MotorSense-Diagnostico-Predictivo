"""
Paridad scratch vs sklearn - prueba de que se entendio el algoritmo.

Tolerancias razonables, no copia exacta: sklearn usa regularizacion
y solver distinto, pero si correlacionan y ROCs estan cerca, esta bien.
"""

import numpy as np
from src.data.dataset import preprocess
from src.models.logistic_scratch import LogisticRegressionScratch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def test_parity_with_sklearn():
    out = preprocess(test_size=0.2, random_state=42)
    X_train, X_test = out["X_train"].values, out["X_test"].values
    y_train, y_test = out["y_train"].values, out["y_test"].values

    # sklearn sin penalizacion para comparar el puro GD vs lbfgs
    try:
        sk = LogisticRegression(max_iter=1000, solver="lbfgs", penalty=None)
        sk.fit(X_train, y_train)
    except Exception:
        sk = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e6)
        sk.fit(X_train, y_train)

    scratch = LogisticRegressionScratch(lr=0.5, n_iter=1500, tol=1e-8)
    scratch.fit(X_train, y_train)

    proba_s = scratch.predict_proba(X_test)[:, 1]
    proba_k = sk.predict_proba(X_test)[:, 1]

    # predicciones muy correlacionadas
    corr = np.corrcoef(proba_s, proba_k)[0, 1]
    assert corr > 0.93, f"corr probas {corr:.3f} muy baja"

    # ROCs cercanos
    roc_s = roc_auc_score(y_test, proba_s)
    roc_k = roc_auc_score(y_test, proba_k)
    assert abs(roc_s - roc_k) < 0.05, f"ROC diff {abs(roc_s-roc_k):.3f} grande"

    # acuerdo en clase > 95%
    pred_s = scratch.predict(X_test)
    pred_k = sk.predict(X_test)
    assert (pred_s == pred_k).mean() > 0.95

    # pesos correlacionados (misma direccion)
    coef_corr = np.corrcoef(scratch.weights.ravel(), sk.coef_.ravel())[0, 1]
    assert coef_corr > 0.85, f"coef corr {coef_corr:.3f} baja"
