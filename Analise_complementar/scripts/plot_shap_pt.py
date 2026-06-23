#!/usr/bin/env python3
"""Regenera a figura de importância SHAP em português, estilo matplotlib enxuto
alinhado às demais figuras do artigo. Lê do CSV de importância já existente,
sem rodar SHAP novamente.

Saída: Artigo/shap_importance.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CSV = PROJECT_ROOT / "Pipes" / "explainability_results" / "shap" / "shap_feature_importance.csv"
OUT = PROJECT_ROOT / "Artigo" / "shap_importance.png"


def main():
    df = pd.read_csv(CSV).head(20).iloc[::-1]  # top 20, ordem crescente (maior em cima no plot)

    plt.figure(figsize=(8, 6))
    plt.barh(df["feature"], df["mean_abs_shap"], edgecolor="black", alpha=0.85)
    plt.xlabel("Importância média (|SHAP|)")
    plt.ylabel("Variável")
    plt.title("Importância das variáveis segundo SHAP: 20 principais")
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Salvo: {OUT}")
    print(df.iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    main()
