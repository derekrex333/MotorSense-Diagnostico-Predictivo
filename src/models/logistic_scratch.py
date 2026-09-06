"""
Regresion logistica desde cero - sin sklearn.

La idea es tener control total del entrenamiento para entender
que pasa con el desbalanceo, y poder comparar luego con sklearn
sin esconder nada debajo del capote.

- sigmoide estable
- log-loss con clipping
- descenso de gradiente batch
- copias defensivas para no mutar lo que entra
"""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoide numericamente estable."""
    # copia defensiva por si z viene de afuera y es vista
    z = np.array(z, dtype=np.float64, copy=True)
    # truco para evitar overflow: dos ramas
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def _log_loss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss binaria promedio."""
    y_true = np.array(y_true, dtype=np.float64, copy=True).ravel()
    y_proba = np.array(y_proba, dtype=np.float64, copy=True).ravel()
    y_proba = np.clip(y_proba, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)))


class LogisticRegressionScratch:
    """
    Regresion logistica con gradiente descendente.

    Parametros bastante familiares si vienes de sklearn,
    pero todo implementado a mano con numpy.

    Ejemplo:
        model = LogisticRegressionScratch(lr=0.1, n_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    """

    def __init__(
        self,
        lr: float = 0.1,
        n_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
        fit_intercept: bool = True,
        random_state: int | None = None,
        class_weight: str | dict | None = None,
    ):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state
        self.class_weight = class_weight

        # se setean en fit
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []
        self.n_features_in_: int | None = None
        self.class_weight_: dict | None = None

    # compatibilidad con codigo que espera .weights / .coef_
    @property
    def weights(self) -> np.ndarray | None:
        return self.weights_

    @property
    def coef_(self) -> np.ndarray | None:
        if self.weights_ is None:
            return None
        return self.weights_.reshape(1, -1)

    @property
    def intercept_(self) -> np.ndarray | None:
        if not self.fit_intercept:
            return np.array([0.0])
        return np.array([self.bias_])

    def fit(self, X, y) -> "LogisticRegressionScratch":
        """
        Entrena con descenso de gradiente.

        Hace copias defensivas de X, y para no modificar
        los arrays originales que vienen de dataset.py.
        """
        # --- copias defensivas desde el inicio ---
        X = np.array(X, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.float64, copy=True).ravel()

        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, llego shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y debe ser 1D, llego shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X ({X.shape[0]}) e y ({y.shape[0]}) no coinciden en filas")
        if not np.all(np.isin(np.unique(y), [0, 1])):
            raise ValueError("y debe ser binaria (0/1)")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # pesos por clase para desbalanceo
        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight == "balanced":
                from collections import Counter

                counter = Counter(y)
                n_classes = len(counter)
                self.class_weight_ = {cls: n_samples / (n_classes * cnt) for cls, cnt in counter.items()}
            elif isinstance(self.class_weight, dict):
                self.class_weight_ = dict(self.class_weight)
            else:
                raise ValueError("class_weight debe ser 'balanced', dict o None")
            sample_weight = np.array([self.class_weight_[int(v)] for v in y], dtype=np.float64)
            # normalizamos para que la media sea 1 y no cambie lr efectivo
            sample_weight = sample_weight / np.mean(sample_weight)
        else:
            self.class_weight_ = None
            sample_weight = np.ones(n_samples, dtype=np.float64)

        # init en ceros - deterministico y simple
        rng = np.random.default_rng(self.random_state)
        # si random_state se da, inicializamos con ruido pequeno para romper simetria
        # si no, ceros (mas estable para comparar con sklearn)
        if self.random_state is not None:
            self.weights_ = rng.normal(0, 0.01, size=n_features).astype(np.float64)
            self.bias_ = float(rng.normal(0, 0.01))
        else:
            self.weights_ = np.zeros(n_features, dtype=np.float64)
            self.bias_ = 0.0

        self.loss_history_ = []
        prev_loss = np.inf

        for i in range(self.n_iter):
            # forward
            z = X @ self.weights_ + (self.bias_ if self.fit_intercept else 0.0)
            proba = _sigmoid(z)
            loss = _log_loss(y, proba)
            self.loss_history_.append(loss)

            # early stopping por tolerancia
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"convergio en iter {i}, loss {loss:.6f}")
                break
            prev_loss = loss

            # gradientes con peso por muestra (si hay desbalanceo)
            error = (proba - y) * sample_weight  # (n,)
            dw = (X.T @ error) / n_samples  # (p,)
            db = float(np.mean(error)) if self.fit_intercept else 0.0

            # update
            self.weights_ -= self.lr * dw
            if self.fit_intercept:
                self.bias_ -= self.lr * db

            if self.verbose and (i % 200 == 0):
                print(f"iter {i:4d} loss {loss:.6f}")

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidades para clase 0 y 1, shape (n, 2)."""
        X = np.array(X, dtype=np.float64, copy=True)
        if self.weights_ is None:
            raise RuntimeError("Modelo no entrenado, llama fit primero")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} features, se entreno con {self.n_features_in_}")

        z = X @ self.weights_ + (self.bias_ if self.fit_intercept else 0.0)
        p1 = _sigmoid(z)
        p0 = 1 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Clase 0/1 segun threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def score(self, X, y) -> float:
        """Accuracy rapida."""
        y = np.array(y, copy=True).ravel()
        return float(np.mean(self.predict(X) == y))
