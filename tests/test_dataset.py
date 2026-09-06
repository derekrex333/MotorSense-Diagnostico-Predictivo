"""
Tests de dataset - lo que no puede fallar antes de entrenar.

Si el split no estratifica o el escalado filtra info de test,
todo lo demas sale mal y no nos enteramos.
"""

import numpy as np
from src.data.dataset import preprocess, build_features, load_raw, SENSOR_COLS


def test_load_raw_shape():
    df = load_raw()
    assert df.shape[0] == 10000
    assert "Machine failure" in df.columns
    assert df.isnull().sum().sum() == 0


def test_build_features_columns():
    df = load_raw()
    X, y = build_features(df)
    # 5 sensores + 2 derivadas + 3 dummies Type = 10
    assert X.shape[1] == 10
    assert len(y) == 10000
    assert set(["Temp_diff", "Power_W", "Type_L", "Type_M", "Type_H"]).issubset(X.columns)
    # no leakage de subtipos
    for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert c not in X.columns


def test_stratified_split_preserves_ratio():
    out = preprocess(test_size=0.2, random_state=42)
    y_train, y_test = out["y_train"], out["y_test"]
    # full es 3.39%, train y test deben estar cerca
    for y in [y_train, y_test]:
        ratio = y.mean()
        assert 0.03 < ratio < 0.04, f"ratio {ratio:.4f} fuera de 3-4% - no estratifico"
    # diferencia entre train y test < 0.5%
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_scaling_only_on_train():
    out = preprocess(test_size=0.2, random_state=42)
    X_train = out["X_train"]
    # media ~0 y std ~1 en train (el scaler se ajusto ahi)
    assert np.allclose(X_train.mean().values, 0, atol=1e-9)
    assert np.allclose(X_train.std().values, 1, atol=1e-2)
    # feature names consistentes
    assert out["feature_names"] == list(X_train.columns)
    assert len(out["feature_names"]) == 10
