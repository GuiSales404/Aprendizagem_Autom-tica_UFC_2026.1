#!/usr/bin/env python3
"""Retreina cada arquitetura iterativa com os melhores hiperparâmetros e
salva o RMSE no conjunto de teste a cada iteração de boosting (ou época,
no caso do MLP). Plota todas as curvas no mesmo gráfico para evidenciar
a convergência.

Saída: Artigo/convergence_plot.png
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, log_evaluation, record_evaluation
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SPLIT = PROJECT_ROOT / "Datasets" / "Split"
TUN = PROJECT_ROOT / "Pipes" / "tuning_results"
OUT = PROJECT_ROOT / "Artigo" / "convergence_plot.png"


def load_data():
    X_train = pd.read_parquet(SPLIT / "X_train_scaled_ohe.parquet")
    X_test = pd.read_parquet(SPLIT / "X_test_scaled_ohe.parquet")
    y_train = pd.read_parquet(SPLIT / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(SPLIT / "y_test.parquet").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def best_params(model_name):
    with open(TUN / model_name / "best_results.json", encoding="utf-8") as f:
        d = json.load(f)
    # remove o hiperparâmetro k_features (que era da etapa SelectKBest)
    p = {k: v for k, v in d["best_params"].items() if k != "k_features"}
    return p


def curve_xgboost(X_tr, y_tr, X_te, y_te):
    p = best_params("xgboost")
    m = XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=-1,
        eval_metric="rmse", **p,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return np.array(m.evals_result()["validation_0"]["rmse"])


def curve_lightgbm(X_tr, y_tr, X_te, y_te):
    p = best_params("lightgbm")
    m = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **p)
    evals = {}
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="rmse",
        callbacks=[record_evaluation(evals), log_evaluation(0)],
    )
    return np.array(evals["valid_0"]["rmse"])


def curve_catboost(X_tr, y_tr, X_te, y_te):
    p = best_params("catboost")
    m = CatBoostRegressor(
        loss_function="RMSE", random_seed=42, verbose=False, **p,
    )
    m.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
    evals = m.get_evals_result()
    return np.array(evals["validation"]["RMSE"])


def curve_mlp(X_tr, y_tr, X_te, y_te, max_iter=300):
    p = best_params("mlp")
    p.pop("max_iter", None)
    m = MLPRegressor(
        max_iter=1, warm_start=True, random_state=42,
        early_stopping=False, **p,
    )
    rmse = []
    for _ in range(max_iter):
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        rmse.append(float(np.sqrt(mean_squared_error(y_te, y_pred))))
    return np.array(rmse)


def main():
    X_tr, X_te, y_tr, y_te = load_data()

    print("Treinando XGBoost...")
    c_xgb = curve_xgboost(X_tr, y_tr, X_te, y_te)
    print("Treinando LightGBM...")
    c_lgb = curve_lightgbm(X_tr, y_tr, X_te, y_te)
    print("Treinando CatBoost...")
    c_cat = curve_catboost(X_tr, y_tr, X_te, y_te)
    print("Treinando MLP...")
    c_mlp = curve_mlp(X_tr, y_tr, X_te, y_te, max_iter=300)

    curves = [
        ("XGBoost",  c_xgb),
        ("LightGBM", c_lgb),
        ("CatBoost", c_cat),
        ("MLP",      c_mlp),
    ]

    plt.figure(figsize=(8, 5))
    for label, c in curves:
        x = np.arange(1, len(c) + 1)
        plt.plot(x, c, linewidth=1.5, label=f"{label} (final = {c[-1]:.4f})")
    plt.xlabel("Iteração de treinamento")
    plt.ylabel("RMSE no conjunto de teste")
    plt.title("Curvas de convergência por arquitetura")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSalvo: {OUT}")
    for label, c in curves:
        print(f"  {label}: inicio={c[0]:.4f}, fim={c[-1]:.4f}, n_iter={len(c)}")


if __name__ == "__main__":
    main()
