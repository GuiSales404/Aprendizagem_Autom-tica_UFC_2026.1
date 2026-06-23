#!/usr/bin/env python3
"""Estatísticas descritivas da variável alvo TAXA_CRE_INT.

Concatena y_train e y_test, calcula estatísticas e gera histograma. Saídas em
Analise_complementar/consolidacao/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "Datasets" / "Split"
DST_DIR = PROJECT_ROOT / "Analise_complementar" / "consolidacao"


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    y_train = pd.read_parquet(SRC_DIR / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(SRC_DIR / "y_test.parquet").iloc[:, 0]
    y_all = pd.concat([y_train, y_test], ignore_index=True)

    def describe(name, s):
        n = len(s)
        zeros = int((s == 0).sum())
        ones = int((s == 1).sum())
        return {
            "split": name,
            "n": n,
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "zeros_count": zeros,
            "zeros_pct": round(100.0 * zeros / n, 2),
            "ones_count": ones,
            "ones_pct": round(100.0 * ones / n, 2),
            "intermediate_pct": round(100.0 * (n - zeros - ones) / n, 2),
        }

    stats = {
        "target": "TAXA_CRE_INT",
        "all": describe("all", y_all),
        "train": describe("train", y_train),
        "test": describe("test", y_test),
    }

    with open(DST_DIR / "target_descriptives.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(8, 5))
    plt.hist(y_all, bins=30, edgecolor="black", alpha=0.85)
    plt.axvline(y_all.mean(), linestyle="--", label=f"média = {y_all.mean():.3f}")
    plt.axvline(y_all.median(), linestyle=":", label=f"mediana = {y_all.median():.3f}")
    plt.xlabel("TAXA_CRE_INT")
    plt.ylabel("Número de municípios")
    plt.title(f"Distribuição da variável alvo — {len(y_all)} municípios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DST_DIR / "target_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
