#!/usr/bin/env python3
"""Gera figura de convergência entre as nove arquiteturas avaliadas, com
intervalos de confiança bootstrap a 95%. Estilo enxuto matplotlib, alinhado
ao target_histogram.png.

Saída: Artigo/convergence_plot.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EVAL_CSV = PROJECT_ROOT / "Pipes" / "evaluation_results" / "evaluation_metrics_with_ci.csv"
LINEAR_JSON = PROJECT_ROOT / "Analise_complementar" / "linear_baseline" / "best_results.json"
OUT = PROJECT_ROOT / "Artigo" / "convergence_plot.png"


DISPLAY_NAMES = {
    "stacking": "Stacking",
    "simple_average": "Voting",
    "xgboost": "XGBoost",
    "random_forest": "Random\nForest",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
    "linear_regression": "Linear",
    "ridge_cv": "Ridge",
}


def main():
    eval_df = pd.read_csv(EVAL_CSV)
    eval_df = eval_df[["model_name", "rmse", "rmse_ci_low", "rmse_ci_high"]].copy()

    with open(LINEAR_JSON, encoding="utf-8") as f:
        linear_data = json.load(f)
    linear_rows = [
        {
            "model_name": r["model_name"],
            "rmse": r["test_rmse"],
            "rmse_ci_low": r["rmse_ci_low"],
            "rmse_ci_high": r["rmse_ci_high"],
        }
        for r in linear_data["results"]
    ]
    linear_df = pd.DataFrame(linear_rows)

    df = pd.concat([eval_df, linear_df], ignore_index=True)
    df = df.sort_values("rmse").reset_index(drop=True)
    df["display"] = df["model_name"].map(DISPLAY_NAMES).fillna(df["model_name"])

    rmse_min = df["rmse"].min()
    rmse_max = df["rmse"].max()
    spread_pct = (rmse_max - rmse_min) / rmse_min * 100

    x = np.arange(len(df))
    yerr_lower = df["rmse"] - df["rmse_ci_low"]
    yerr_upper = df["rmse_ci_high"] - df["rmse"]

    plt.figure(figsize=(8, 5))
    plt.bar(x, df["rmse"], edgecolor="black", alpha=0.85,
            yerr=[yerr_lower, yerr_upper], capsize=4,
            ecolor="black", error_kw={"elinewidth": 1.0})

    plt.axhline(rmse_min, linestyle="--",
                label=f"melhor RMSE = {rmse_min:.4f}")
    plt.axhline(rmse_max, linestyle=":",
                label=f"pior RMSE = {rmse_max:.4f}")

    plt.xticks(x, df["display"], rotation=0)
    plt.xlabel("Arquitetura")
    plt.ylabel("RMSE")
    plt.title(f"Convergência entre arquiteturas: spread de {spread_pct:.1f}%")
    plt.ylim(df["rmse_ci_low"].min() - 0.005, df["rmse_ci_high"].max() + 0.005)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Salvo: {OUT}")
    print(f"Spread: {spread_pct:.2f}%")
    print(df[["display", "rmse", "rmse_ci_low", "rmse_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
