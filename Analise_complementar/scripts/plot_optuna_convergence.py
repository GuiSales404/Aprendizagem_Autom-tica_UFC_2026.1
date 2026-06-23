#!/usr/bin/env python3
"""Gera curvas de convergência do Optuna para cada arquitetura tunada:
o melhor RMSE de validação cruzada acumulado a cada trial (best-so-far),
todas sobrepostas no mesmo gráfico para evidenciar convergência.

Saída: Artigo/convergence_plot.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TUNING_DIR = PROJECT_ROOT / "Pipes" / "tuning_results"
OUT = PROJECT_ROOT / "Artigo" / "convergence_plot.png"


MODELS = [
    ("xgboost",       "XGBoost"),
    ("lightgbm",      "LightGBM"),
    ("catboost",      "CatBoost"),
    ("random_forest", "Random Forest"),
    ("mlp",           "MLP"),
]


def best_so_far(values):
    """Retorna o vetor de mínimo acumulado."""
    return np.minimum.accumulate(values)


def main():
    plt.figure(figsize=(8, 5))

    final_values = {}
    for folder, label in MODELS:
        csv = TUNING_DIR / folder / "all_trials.csv"
        df = pd.read_csv(csv).sort_values("number").reset_index(drop=True)
        bsf = best_so_far(df["value"].values)
        trials = df["number"].values + 1  # 1-indexado para o gráfico
        plt.plot(trials, bsf, marker="o", markersize=3, linewidth=1.5, label=label)
        final_values[label] = bsf[-1]

    plt.xlabel("Trial de otimização (Optuna)")
    plt.ylabel("Melhor RMSE de validação cruzada acumulado")
    plt.title("Convergência do tuning: best-so-far por arquitetura")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Salvo: {OUT}")
    print("RMSE final (best-so-far) por modelo:")
    for k, v in final_values.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
