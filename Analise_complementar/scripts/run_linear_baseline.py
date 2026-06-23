#!/usr/bin/env python3
"""Baseline linear (LinearRegression + RidgeCV) para comparar com os modelos
gradient boosting / RF / MLP da Tabela V do artigo.

Reproduz o pré-processamento de Pipes/train_individual_models.py: imputação
mediana + StandardScaler nos numéricos + OneHotEncoder em SG_UF. Calcula RMSE,
MAE, R² no test set e IC95% via bootstrap.

Saídas em Analise_complementar/linear_baseline/.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "Datasets" / "Split"
DST_DIR = PROJECT_ROOT / "Analise_complementar" / "linear_baseline"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_columns(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(X):
    numeric_cols, categorical_cols = get_columns(X)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(metric_fn(y_true[idx], y_pred[idx]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def evaluate(name, model, X_train, X_test, y_train, y_test):
    preprocessor = make_preprocessor(X_train)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    m = metrics_dict(y_test, y_pred)
    ci_rmse = bootstrap_ci(y_test, y_pred, rmse)
    ci_mae = bootstrap_ci(y_test, y_pred, lambda yt, yp: float(mean_absolute_error(yt, yp)))
    ci_r2 = bootstrap_ci(y_test, y_pred, lambda yt, yp: float(r2_score(yt, yp)))

    return {
        "model_name": name,
        "test_rmse": m["rmse"],
        "test_mae": m["mae"],
        "test_r2": m["r2"],
        "rmse_ci_low": ci_rmse["ci_low"],
        "rmse_ci_high": ci_rmse["ci_high"],
        "mae_ci_low": ci_mae["ci_low"],
        "mae_ci_high": ci_mae["ci_high"],
        "r2_ci_low": ci_r2["ci_low"],
        "r2_ci_high": ci_r2["ci_high"],
        "y_pred": y_pred,
    }


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_parquet(SRC_DIR / "X_train_raw.parquet")
    X_test = pd.read_parquet(SRC_DIR / "X_test_raw.parquet")
    y_train = pd.read_parquet(SRC_DIR / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(SRC_DIR / "y_test.parquet").iloc[:, 0]

    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    results = []
    pred_cols = {"y_true": y_test.values}

    for name, model in [
        ("linear_regression", LinearRegression()),
        ("ridge_cv", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ]:
        r = evaluate(name, model, X_train, X_test, y_train, y_test)
        pred_cols[name] = r.pop("y_pred")
        results.append(r)
        print(
            f"{name}: RMSE={r['test_rmse']:.4f} "
            f"[{r['rmse_ci_low']:.4f}; {r['rmse_ci_high']:.4f}] "
            f"MAE={r['test_mae']:.4f} R²={r['test_r2']:.4f}"
        )

    pd.DataFrame(pred_cols).to_csv(DST_DIR / "test_predictions.csv", index=False)
    with open(DST_DIR / "best_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "n_bootstrap": N_BOOTSTRAP}, f, indent=2)


if __name__ == "__main__":
    main()
