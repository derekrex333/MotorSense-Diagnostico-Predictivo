"""
Mapeo feature -> peso con nombres reales de sensores.

La gracia de la logistica es que puedes leer los pesos:
positivo empuja hacia falla, negativo hacia no falla.
Aqui lo dejamos en lenguaje de taller, no solo numeros.
"""

import numpy as np
import pandas as pd


# nombres mas amigables para reportes, manteniendo el original para trazar
SENSOR_LABELS = {
    "Air temperature [K]": "Temp. aire",
    "Process temperature [K]": "Temp. proceso",
    "Rotational speed [rpm]": "Velocidad",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Desgaste",
    "Temp_diff": "dT (proceso-aire)",
    "Power_W": "Potencia",
    "Type_H": "Producto H",
    "Type_L": "Producto L",
    "Type_M": "Producto M",
}


def get_feature_weights(model, feature_names: list[str]) -> dict[str, float]:
    """
    Extrae pesos del modelo (scratch o sklearn) y los mapea a nombres reales.
    Hace copias defensivas para no mutar el modelo.
    """
    feature_names = list(feature_names)  # copia

    if hasattr(model, "weights") and model.weights is not None:
        # scratch
        coef = np.array(model.weights, copy=True).ravel()
    elif hasattr(model, "coef_") and model.coef_ is not None:
        coef = np.array(model.coef_, copy=True).ravel()
    else:
        raise AttributeError("Modelo no expone coeficientes (weights/coef_)")

    if len(coef) != len(feature_names):
        raise ValueError(f"coef {len(coef)} != features {len(feature_names)}")

    return dict(zip(feature_names, coef.astype(float)))


def sort_by_importance(feature_weights: dict[str, float]) -> list[tuple[str, float]]:
    """Ordena por |peso| descendente - lo que mas mueve la prediccion."""
    items = list(feature_weights.items())
    return sorted(items, key=lambda x: abs(x[1]), reverse=True)


def to_dataframe(feature_weights: dict[str, float], sort: bool = True) -> pd.DataFrame:
    """Convierte a DataFrame con etiqueta amigable y signo."""
    items = sort_by_importance(feature_weights) if sort else list(feature_weights.items())
    rows = []
    for feat, w in items:
        rows.append({
            "feature": feat,
            "sensor": SENSOR_LABELS.get(feat, feat),
            "peso": float(w),
            "abs_peso": abs(float(w)),
            "direccion": "-> falla" if w > 0 else "-> no falla",
        })
    return pd.DataFrame(rows)


def mechanical_contrast(feature_weights: dict[str, float]) -> str:
    """
    Contraste rapido si el orden tiene sentido mecanico.
    Basado en lo que vimos en la exploracion y sentido comun de taller.
    """
    sorted_feats = [f for f, _ in sort_by_importance(feature_weights)]
    top3 = sorted_feats[:3]

    notes = []
    # Torque y velocidad son clasicos de sobrecarga
    if "Torque [Nm]" in top3 or "Power_W" in top3:
        notes.append("Torque/Potencia arriba: cuadra, sobrecarga rompe.")
    if "Tool wear [min]" in top3:
        notes.append("Desgaste arriba: logico, herramienta gastada falla mas.")
    if "Rotational speed [rpm]" in sorted_feats and feature_weights.get("Rotational speed [rpm]", 0) > 0:
        # en exploracion velocidad media era menor en falla, pero con torque alto la potencia sube
        notes.append("Velocidad con peso positivo: ojo, en datos crudos era menor en falla, pero aqui controla por torque (Power).")
    if "Type_L" in top3:
        notes.append("Type_L arriba: coincide con que L falla mas (3.9%).")
    if not notes:
        notes.append("Orden no obvio, revisar si hay leakage o escala mal.")

    return " ".join(notes)
