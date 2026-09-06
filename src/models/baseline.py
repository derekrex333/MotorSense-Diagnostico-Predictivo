"""
Baselines con scikit-learn para comparar directo con la logistica desde cero.

Usamos exactamente el mismo split y escalado que en dataset.py,
asi la comparacion es limpia. Dejo dos helpers simples por si
queremos probar class_weight mas adelante.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression(**kwargs) -> LogisticRegression:
    """
    LogisticRegression con defaults sensatos para este dataset.

    Por defecto usa lbfgs y 1000 iteraciones, sin class_weight
    para comparar mano a mano con el scratch. Si quieres balancear,
    pasa class_weight='balanced'.
    """
    defaults = dict(
        max_iter=1000,
        solver="lbfgs",
        # penalty="l2" es el default, lo dejamos asi
    )
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def get_random_forest(**kwargs) -> RandomForestClassifier:
    """
    RandomForest baseline. Con pocos arboles ya separa bien
    por el desbalanceo, pero no es lineal y no nos deja
    interpretar coeficientes como la logistica.
    """
    defaults = dict(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def train_baselines(X_train, y_train, **lr_kwargs):
    """Entrena LR y RF y los devuelve en un dict."""
    lr = get_logistic_regression(**lr_kwargs)
    rf = get_random_forest()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return {"logreg": lr, "forest": rf}
