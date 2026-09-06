"""
Carga, preprocesado y split estratificado para AI4I 2020.

Ideas simples que encontramos en la exploracion:
- target muy desbalanceado (3.4% falla) -> split con stratify
- sensores en escalas distintas -> StandardScaler ajustado solo en train
- Type es categorico -> one-hot
- dos features derivadas que ayudan: Temp_diff y Power
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# columnas que vienen del CSV original
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COL = "Machine failure"
TYPE_COL = "Type"


def load_raw(path: str | Path = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    """Carga el CSV tal cual esta en data/raw."""
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Arma X, y limpios:
    - saca IDs (UDI, Product ID) y subtipos de falla (TWF...RNF) que no usamos para el baseline
    - crea Temp_diff y Power
    - one-hot para Type
    """
    df = df.copy()

    # features derivadas - las vimos en la exploracion, aportan separacion
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60

    # one-hot Type (L, M, H)
    df = pd.get_dummies(df, columns=[TYPE_COL], prefix="Type", dtype=float)

    # columnas a usar como X
    feature_cols = SENSOR_COLS + ["Temp_diff", "Power_W"] + [c for c in df.columns if c.startswith("Type_")]

    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split que preserva el 3.4% de fallas en train y test."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Ajusta StandardScaler solo con train (para no filtrar info de test)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def preprocess(
    path: str | Path = "data/raw/ai4i2020.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Pipeline completo: carga -> features -> split -> normalizacion.
    Devuelve dict con todo lo necesario para entrenar.
    """
    df = load_raw(path)
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size, random_state)

    scaler = fit_scaler(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }


if __name__ == "__main__":
    out = preprocess()
    print("Feature names:", out["feature_names"])
    print("X_train:", out["X_train"].shape, "y_train falla:", out["y_train"].mean())
    print("X_test:", out["X_test"].shape, "y_test falla:", out["y_test"].mean())
    print(out["X_train"].describe().round(2).to_string())
