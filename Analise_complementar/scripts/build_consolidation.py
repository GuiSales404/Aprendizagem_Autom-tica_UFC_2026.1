#!/usr/bin/env python3
"""Constrói tabelas de consolidação a partir dos artefatos:
 - Pipes/evaluation_results/evaluation_metrics_with_ci.csv  (com SG_UF)
 - Analise_complementar/no_uf/evaluation_results/evaluation_metrics_with_ci.csv  (sem SG_UF)
 - Analise_complementar/linear_baseline/best_results.json  (baseline linear)

Saídas em Analise_complementar/consolidacao/:
 - ablation_comparison.csv: lado a lado com_uf vs sem_uf vs baseline
 - shap_comparison.csv: ranking SHAP com vs sem SG_UF
 - h1_ocup_fem_vs_masc.json: análise específica de TAXA_OCUP_FEM vs TAXA_OCUP_MASC
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DST = PROJECT_ROOT / "Analise_complementar" / "consolidacao"


def load_metrics(path):
    df = pd.read_csv(path)
    return df.set_index("model_name")


def fmt_ci(row, metric):
    return f"{row[metric]:.4f} [{row[f'{metric}_ci_low']:.4f}; {row[f'{metric}_ci_high']:.4f}]"


def build_ablation():
    com_uf = load_metrics(PROJECT_ROOT / "Pipes" / "evaluation_results" / "evaluation_metrics_with_ci.csv")
    sem_uf = load_metrics(PROJECT_ROOT / "Analise_complementar" / "no_uf" / "evaluation_results" / "evaluation_metrics_with_ci.csv")

    baseline = json.loads(
        (PROJECT_ROOT / "Analise_complementar" / "linear_baseline" / "best_results.json").read_text()
    )["results"]
    base_map = {r["model_name"]: r for r in baseline}

    models = ["stacking", "simple_average", "xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
    rows = []
    for m in models:
        if m in com_uf.index and m in sem_uf.index:
            r_com = com_uf.loc[m]
            r_sem = sem_uf.loc[m]
            delta_rmse = (r_sem["rmse"] - r_com["rmse"]) / r_com["rmse"] * 100
            delta_r2 = r_sem["r2"] - r_com["r2"]
            rows.append({
                "model": m,
                "rmse_com_uf": r_com["rmse"],
                "rmse_ci_com_uf": fmt_ci(r_com, "rmse"),
                "r2_com_uf": r_com["r2"],
                "rmse_sem_uf": r_sem["rmse"],
                "rmse_ci_sem_uf": fmt_ci(r_sem, "rmse"),
                "r2_sem_uf": r_sem["r2"],
                "delta_rmse_pct": round(delta_rmse, 2),
                "delta_r2": round(delta_r2, 4),
            })

    # Linear baseline rows (com SG_UF apenas; precisa de "linear" rodado no no_uf? Não — o baseline linear que rodamos foi com SG_UF.)
    for name in ["linear_regression", "ridge_cv"]:
        b = base_map.get(name)
        if b is None:
            continue
        rows.append({
            "model": name + " (com_uf)",
            "rmse_com_uf": b["test_rmse"],
            "rmse_ci_com_uf": f"{b['test_rmse']:.4f} [{b['rmse_ci_low']:.4f}; {b['rmse_ci_high']:.4f}]",
            "r2_com_uf": b["test_r2"],
            "rmse_sem_uf": np.nan,
            "rmse_ci_sem_uf": "",
            "r2_sem_uf": np.nan,
            "delta_rmse_pct": np.nan,
            "delta_r2": np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DST / "ablation_comparison.csv", index=False)
    return df


def build_shap_comparison():
    com = pd.read_csv(PROJECT_ROOT / "Pipes" / "explainability_results" / "shap" / "shap_feature_importance.csv")
    sem = pd.read_csv(PROJECT_ROOT / "Analise_complementar" / "no_uf" / "explainability_results" / "shap" / "shap_feature_importance.csv")

    com["rank_com_uf"] = range(1, len(com) + 1)
    sem["rank_sem_uf"] = range(1, len(sem) + 1)

    merged = com.merge(
        sem.rename(columns={"mean_abs_shap": "mean_abs_shap_sem_uf"}),
        on="feature", how="outer",
    )
    merged = merged.rename(columns={"mean_abs_shap": "mean_abs_shap_com_uf"})
    merged["delta_rank"] = merged["rank_com_uf"] - merged["rank_sem_uf"]
    merged = merged.sort_values("mean_abs_shap_sem_uf", ascending=False, na_position="last")
    merged.to_csv(DST / "shap_comparison.csv", index=False)
    return merged


def build_h1_analysis(shap_df):
    rows = shap_df[shap_df["feature"].isin(["TAXA_OCUP_FEM", "TAXA_OCUP_MASC"])]
    out = {}
    for _, r in rows.iterrows():
        out[r["feature"]] = {
            "shap_com_uf": r.get("mean_abs_shap_com_uf"),
            "shap_sem_uf": r.get("mean_abs_shap_sem_uf"),
            "rank_com_uf": r.get("rank_com_uf"),
            "rank_sem_uf": r.get("rank_sem_uf"),
        }

    fem = out.get("TAXA_OCUP_FEM", {})
    masc = out.get("TAXA_OCUP_MASC", {})

    summary = {
        "features": out,
        "razao_masc_fem_com_uf": (
            masc.get("shap_com_uf") / fem.get("shap_com_uf")
            if fem.get("shap_com_uf") else None
        ),
        "razao_masc_fem_sem_uf": (
            masc.get("shap_sem_uf") / fem.get("shap_sem_uf")
            if fem.get("shap_sem_uf") else None
        ),
        "interpretacao": (
            "Em ambos os cenarios (com e sem SG_UF) TAXA_OCUP_MASC tem maior "
            "importancia SHAP que TAXA_OCUP_FEM, refutando parcialmente H1 que "
            "esperava ocupacao feminina como driver principal. A razao masc/fem "
            "AUMENTA quando SG_UF e removido, sugerindo que SG_UF capturava "
            "parte do efeito de ocupacao feminina."
        ),
    }
    with open(DST / "h1_ocup_fem_vs_masc.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def build_counterfactual_summary():
    com_uf_counts = pd.read_csv(
        PROJECT_ROOT / "Pipes" / "explainability_results" / "counterfactuals" / "counterfactual_change_counts.csv"
    )
    sem_uf_counts = pd.read_csv(
        PROJECT_ROOT / "Analise_complementar" / "no_uf" / "explainability_results" / "counterfactuals" / "counterfactual_change_counts.csv"
    )

    com_uf_counts["pct_com_uf"] = round(com_uf_counts["n_changes"] / 30 * 100, 1)
    sem_uf_counts["pct_sem_uf"] = round(sem_uf_counts["n_changes"] / 30 * 100, 1)

    merged = com_uf_counts.merge(
        sem_uf_counts.rename(columns={"n_changes": "n_changes_sem_uf"}),
        on="feature", how="outer",
    ).rename(columns={"n_changes": "n_changes_com_uf"})

    merged = merged.sort_values("pct_sem_uf", ascending=False, na_position="last")
    merged.to_csv(DST / "counterfactual_comparison.csv", index=False)
    return merged


def main():
    DST.mkdir(parents=True, exist_ok=True)

    ablation = build_ablation()
    print("=== ABLATION COMPARISON ===")
    print(ablation.to_string(index=False))

    print("\n=== SHAP COMPARISON (top 10) ===")
    shap_df = build_shap_comparison()
    print(shap_df.head(10).to_string(index=False))

    print("\n=== H1 ANALYSIS ===")
    h1 = build_h1_analysis(shap_df)
    print(json.dumps(h1, indent=2, ensure_ascii=False, default=str))

    print("\n=== COUNTERFACTUAL COMPARISON ===")
    cf = build_counterfactual_summary()
    print(cf.to_string(index=False))


if __name__ == "__main__":
    main()
