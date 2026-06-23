#!/usr/bin/env python3
"""Gera a figura de contagem de alterações nas explicações contrafactuais
via DiCE em português, estilo matplotlib enxuto alinhado às demais figuras.

Saída: Artigo/counterfactual_changes.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CSV = PROJECT_ROOT / "Pipes" / "explainability_results" / "counterfactuals" / "counterfactual_change_counts.csv"
OUT = PROJECT_ROOT / "Artigo" / "counterfactual_changes.png"

TOTAL_CFS = 30


def main():
    df = pd.read_csv(CSV)
    df = df.sort_values("n_changes", ascending=True)
    df["pct"] = df["n_changes"] / TOTAL_CFS * 100

    plt.figure(figsize=(8, max(4, 0.35 * len(df))))
    plt.barh(df["feature"], df["pct"], edgecolor="black", alpha=0.85)
    plt.xlabel("Frequência de alteração (\\% dos contrafactuais)")
    plt.ylabel("Variável")
    plt.title("Alterações mais frequentes nos contrafactuais DiCE")
    plt.xlim(0, max(df["pct"].max() * 1.15, 10))
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Salvo: {OUT}")
    print(df.iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    main()
