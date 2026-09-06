"""
Manejo de desbalanceo - intercambiable entre class_weight y SMOTE.

La idea es no atarnos a una sola tecnica: si tienes pocos datos
de falla, ponderas; si quieres sintetizar, usas SMOTE. Ambas estan
aqui y se pueden cambiar sin tocar el modelo.
"""

import numpy as np
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


def get_class_weights(y, method: str = "balanced") -> dict:
    """
    Calcula pesos por clase al estilo sklearn 'balanced':
    w_j = n_samples / (n_classes * count_j)
    """
    y = np.array(y, copy=True).ravel()
    counter = Counter(y)
    n_samples = len(y)
    n_classes = len(counter)

    if method != "balanced":
        raise ValueError("solo method='balanced' por ahora")

    weights = {cls: n_samples / (n_classes * cnt) for cls, cnt in counter.items()}
    return weights


def get_sample_weights(y, class_weights: dict) -> np.ndarray:
    """Expande pesos de clase a pesos por muestra."""
    y = np.array(y, copy=True).ravel()
    return np.array([class_weights[int(v)] for v in y], dtype=np.float64)


def apply_smote(X, y, random_state: int = 42):
    """
    Aplica SMOTE para balancear. Si imbalanced-learn no esta,
    avisa y no rompe el flujo.
    """
    if SMOTE is None:
        raise ImportError("imblearn no instalado: pip install imbalanced-learn")

    X = np.array(X, copy=True)
    y = np.array(y, copy=True).ravel()

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
