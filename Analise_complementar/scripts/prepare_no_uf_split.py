#!/usr/bin/env python3
"""Gera split paralelo sem a coluna SG_UF.

Lê os parquets originais em Datasets/Split/ (apenas leitura) e escreve cópias
sem SG_UF em Analise_complementar/data/. Os scripts originais em Pipes/ podem
então ser invocados apontando --splits-dir para essa nova pasta.
"""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "Datasets" / "Split"
DST_DIR = PROJECT_ROOT / "Analise_complementar" / "data"

COL_TO_DROP = "SG_UF"


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_parquet(SRC_DIR / "X_train_raw.parquet")
    X_test = pd.read_parquet(SRC_DIR / "X_test_raw.parquet")
    y_train = pd.read_parquet(SRC_DIR / "y_train.parquet")
    y_test = pd.read_parquet(SRC_DIR / "y_test.parquet")

    print(f"Original: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Colunas: {list(X_train.columns)}")

    if COL_TO_DROP not in X_train.columns:
        raise SystemExit(f"Coluna {COL_TO_DROP} não encontrada em X_train_raw.")

    X_train_no_uf = X_train.drop(columns=[COL_TO_DROP])
    X_test_no_uf = X_test.drop(columns=[COL_TO_DROP])

    X_train_no_uf.to_parquet(DST_DIR / "X_train_raw.parquet", index=False)
    X_test_no_uf.to_parquet(DST_DIR / "X_test_raw.parquet", index=False)
    y_train.to_parquet(DST_DIR / "y_train.parquet", index=False)
    y_test.to_parquet(DST_DIR / "y_test.parquet", index=False)

    print(f"\nSem SG_UF: X_train {X_train_no_uf.shape}, X_test {X_test_no_uf.shape}")
    print(f"Escrito em: {DST_DIR}")


if __name__ == "__main__":
    main()
