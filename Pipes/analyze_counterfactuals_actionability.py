#!/usr/bin/env python3
# analyze_counterfactuals_actionability.py

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_NON_ACTIONABLE = [
    "SG_UF",
    "UF",
    "ESTADO",
    "COD_UF",
    "COD_MUNICIPIO",
    "CD_MUN",
    "NM_MUN",
    "MUNICIPIO",
    "ID",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze counterfactual explanations separating actionable and "
            "non-actionable features."
        )
    )

    parser.add_argument(
        "--explainability-dir",
        default="explainability_results",
        help="Path to explainability_results directory.",
    )

    parser.add_argument(
        "--zip-path",
        default=None,
        help="Optional zip file containing explainability_results.",
    )

    parser.add_argument(
        "--output-dir",
        default="counterfactual_actionability_analysis",
    )

    parser.add_argument(
        "--non-actionable-features",
        nargs="*",
        default=DEFAULT_NON_ACTIONABLE,
        help="Feature names considered non-actionable.",
    )

    parser.add_argument(
        "--actionable-features",
        nargs="*",
        default=None,
        help=(
            "Optional explicit actionable feature list. If omitted, all changed "
            "features not in non-actionable-features are treated as actionable."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
    )

    return parser.parse_args()


def maybe_extract_zip(zip_path, output_dir):
    zip_path = Path(zip_path)
    extract_dir = Path(output_dir) / "_unzipped_explainability"

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    candidates = list(extract_dir.rglob("counterfactual_results.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find counterfactual_results.csv inside the zip."
        )

    return candidates[0].parents[0].parents[0]


def find_counterfactual_files(explainability_dir):
    explainability_dir = Path(explainability_dir)

    cf_results_candidates = list(explainability_dir.rglob("counterfactual_results.csv"))
    cf_changes_candidates = list(explainability_dir.rglob("counterfactual_feature_changes.csv"))

    if not cf_results_candidates:
        raise FileNotFoundError(
            f"counterfactual_results.csv not found under {explainability_dir}"
        )

    if not cf_changes_candidates:
        raise FileNotFoundError(
            f"counterfactual_feature_changes.csv not found under {explainability_dir}"
        )

    return cf_results_candidates[0], cf_changes_candidates[0]


def classify_feature(feature, actionable_features, non_actionable_features):
    if actionable_features is not None:
        return "actionable" if feature in actionable_features else "non_actionable"

    return "non_actionable" if feature in non_actionable_features else "actionable"


def safe_numeric_delta(original, counterfactual):
    try:
        o = float(original)
        c = float(counterfactual)
        return c - o
    except Exception:
        return np.nan


def build_change_analysis(changes_df, actionable_features, non_actionable_features):
    df = changes_df.copy()

    df["feature_type"] = df["feature"].apply(
        lambda f: classify_feature(
            f,
            actionable_features=actionable_features,
            non_actionable_features=non_actionable_features,
        )
    )

    df["numeric_delta"] = df.apply(
        lambda row: safe_numeric_delta(row["original_value"], row["counterfactual_value"]),
        axis=1,
    )

    df["changed_from_to"] = (
        df["original_value"].astype(str)
        + " → "
        + df["counterfactual_value"].astype(str)
    )

    return df


def build_cf_level_summary(results_df, changes_df):
    group_cols = ["original_index", "counterfactual_id"]

    summary = (
        changes_df
        .groupby(group_cols)
        .agg(
            n_total_changes=("feature", "count"),
            n_actionable_changes=("feature_type", lambda s: int((s == "actionable").sum())),
            n_non_actionable_changes=("feature_type", lambda s: int((s == "non_actionable").sum())),
            actionable_features=("feature", lambda s: ", ".join(sorted(set(
                changes_df.loc[s.index][changes_df.loc[s.index, "feature_type"] == "actionable"]["feature"]
            )))),
            non_actionable_features=("feature", lambda s: ", ".join(sorted(set(
                changes_df.loc[s.index][changes_df.loc[s.index, "feature_type"] == "non_actionable"]["feature"]
            )))),
        )
        .reset_index()
    )

    keep_cols = [
        "original_index",
        "counterfactual_id",
        "original_prediction",
        "counterfactual_prediction",
        "desired_low",
        "desired_high",
    ]

    available = [c for c in keep_cols if c in results_df.columns]

    merged = results_df[available].merge(
        summary,
        on=["original_index", "counterfactual_id"],
        how="left",
    )

    merged["n_total_changes"] = merged["n_total_changes"].fillna(0).astype(int)
    merged["n_actionable_changes"] = merged["n_actionable_changes"].fillna(0).astype(int)
    merged["n_non_actionable_changes"] = merged["n_non_actionable_changes"].fillna(0).astype(int)

    merged["prediction_delta"] = (
        merged["counterfactual_prediction"] - merged["original_prediction"]
    )

    merged["abs_prediction_delta"] = merged["prediction_delta"].abs()

    merged["only_actionable"] = (
        (merged["n_actionable_changes"] > 0)
        & (merged["n_non_actionable_changes"] == 0)
    )

    merged["uses_non_actionable"] = merged["n_non_actionable_changes"] > 0

    return merged


def plot_bar(df, x_col, y_col, title, xlabel, ylabel, path, top_n=20):
    plot_df = df.head(top_n).copy()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")

    for i, row in enumerate(plot_df.itertuples()):
        value = getattr(row, y_col)
        plt.text(i, value, str(value), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stacked_actionability(cf_summary, output_dir):
    counts = pd.Series({
        "Only actionable": int(cf_summary["only_actionable"].sum()),
        "Uses non-actionable": int(cf_summary["uses_non_actionable"].sum()),
    })

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Number of counterfactuals")
    plt.title("Counterfactual actionability profile")

    for i, value in enumerate(counts.values):
        plt.text(i, value, str(value), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "counterfactual_actionability_profile.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_changes_distribution(cf_summary, output_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(cf_summary["n_total_changes"], bins=range(0, cf_summary["n_total_changes"].max() + 2), alpha=0.8, edgecolor="black")
    plt.xlabel("Number of changed features")
    plt.ylabel("Number of counterfactuals")
    plt.title("Sparsity of counterfactual explanations")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "counterfactual_sparsity_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_report(output_dir, cf_summary, feature_counts, type_counts, non_actionable_features, actionable_features):
    n_cf = len(cf_summary)
    n_only_actionable = int(cf_summary["only_actionable"].sum())
    n_uses_non_actionable = int(cf_summary["uses_non_actionable"].sum())

    pct_only_actionable = 100 * n_only_actionable / n_cf if n_cf else 0
    pct_uses_non_actionable = 100 * n_uses_non_actionable / n_cf if n_cf else 0

    mean_changes = cf_summary["n_total_changes"].mean()
    mean_actionable = cf_summary["n_actionable_changes"].mean()
    mean_non_actionable = cf_summary["n_non_actionable_changes"].mean()

    top_feature = feature_counts.iloc[0]["feature"] if len(feature_counts) else "N/A"
    top_feature_count = int(feature_counts.iloc[0]["n_changes"]) if len(feature_counts) else 0

    report = []
    report.append("# Counterfactual Actionability Analysis\n")
    report.append("## Configuration\n")
    report.append(f"- Non-actionable features: `{non_actionable_features}`")
    if actionable_features is not None:
        report.append(f"- Explicit actionable features: `{actionable_features}`")
    else:
        report.append("- Actionable features: all changed features not listed as non-actionable.")

    report.append("\n## Summary\n")
    report.append(f"- Total counterfactuals analyzed: **{n_cf}**")
    report.append(f"- Counterfactuals using only actionable features: **{n_only_actionable}** ({pct_only_actionable:.1f}%)")
    report.append(f"- Counterfactuals using at least one non-actionable feature: **{n_uses_non_actionable}** ({pct_uses_non_actionable:.1f}%)")
    report.append(f"- Mean number of changed features per counterfactual: **{mean_changes:.2f}**")
    report.append(f"- Mean actionable changes per counterfactual: **{mean_actionable:.2f}**")
    report.append(f"- Mean non-actionable changes per counterfactual: **{mean_non_actionable:.2f}**")
    report.append(f"- Most frequently changed feature: **{top_feature}** ({top_feature_count} changes)")

    report.append("\n## Feature change frequency\n")
    report.append(feature_counts.head(20).to_markdown(index=False))

    report.append("\n\n## Change frequency by actionability type\n")
    report.append(type_counts.to_markdown(index=False))

    report.append("\n\n## Interpretation guide\n")
    report.append(
        "- If many counterfactuals rely on non-actionable features, the model may be using useful predictive proxies that are weak for intervention-oriented explanations."
    )
    report.append(
        "- If actionable features frequently appear with small sparse changes, the explanations are more useful for practical or policy interpretation."
    )
    report.append(
        "- A strong dependence on geographic or identity-like variables should be treated as a potential proxy/bias issue, even when predictive performance is good."
    )

    report.append("\n\n## Generated files\n")
    for p in sorted(Path(output_dir).glob("*")):
        if p.name != "counterfactual_actionability_report.md":
            report.append(f"- `{p.name}`")

    Path(output_dir, "counterfactual_actionability_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_path:
        explainability_dir = maybe_extract_zip(args.zip_path, output_dir)
    else:
        explainability_dir = Path(args.explainability_dir)

    cf_results_path, cf_changes_path = find_counterfactual_files(explainability_dir)

    results_df = pd.read_csv(cf_results_path)
    changes_df = pd.read_csv(cf_changes_path)

    actionable_features = set(args.actionable_features) if args.actionable_features else None
    non_actionable_features = set(args.non_actionable_features)

    changes_labeled = build_change_analysis(
        changes_df=changes_df,
        actionable_features=actionable_features,
        non_actionable_features=non_actionable_features,
    )

    cf_summary = build_cf_level_summary(
        results_df=results_df,
        changes_df=changes_labeled,
    )

    feature_counts = (
        changes_labeled
        .groupby(["feature", "feature_type"])
        .size()
        .reset_index(name="n_changes")
        .sort_values("n_changes", ascending=False)
    )

    type_counts = (
        changes_labeled
        .groupby("feature_type")
        .size()
        .reset_index(name="n_changes")
        .sort_values("n_changes", ascending=False)
    )

    numeric_deltas = (
        changes_labeled
        .dropna(subset=["numeric_delta"])
        .groupby(["feature", "feature_type"])
        .agg(
            n_changes=("numeric_delta", "count"),
            mean_delta=("numeric_delta", "mean"),
            median_delta=("numeric_delta", "median"),
            mean_abs_delta=("numeric_delta", lambda s: float(np.mean(np.abs(s)))),
        )
        .reset_index()
        .sort_values("n_changes", ascending=False)
    )

    changes_labeled.to_csv(output_dir / "counterfactual_feature_changes_labeled.csv", index=False)
    cf_summary.to_csv(output_dir / "counterfactual_level_actionability_summary.csv", index=False)
    feature_counts.to_csv(output_dir / "counterfactual_feature_change_counts_by_type.csv", index=False)
    type_counts.to_csv(output_dir / "counterfactual_change_counts_by_actionability.csv", index=False)
    numeric_deltas.to_csv(output_dir / "counterfactual_numeric_delta_summary.csv", index=False)

    plot_bar(
        df=feature_counts,
        x_col="feature",
        y_col="n_changes",
        title="Most frequently changed counterfactual features",
        xlabel="Feature",
        ylabel="Number of changes",
        path=output_dir / "top_changed_features_by_actionability.png",
        top_n=args.top_n,
    )

    plot_stacked_actionability(cf_summary, output_dir)
    plot_changes_distribution(cf_summary, output_dir)

    if len(numeric_deltas) > 0:
        plot_bar(
            df=numeric_deltas.sort_values("mean_abs_delta", ascending=False),
            x_col="feature",
            y_col="mean_abs_delta",
            title="Largest average absolute numeric changes",
            xlabel="Feature",
            ylabel="Mean absolute delta",
            path=output_dir / "top_numeric_mean_abs_delta.png",
            top_n=args.top_n,
        )

    write_report(
        output_dir=output_dir,
        cf_summary=cf_summary,
        feature_counts=feature_counts,
        type_counts=type_counts,
        non_actionable_features=sorted(non_actionable_features),
        actionable_features=sorted(actionable_features) if actionable_features else None,
    )

    print("\nCounterfactual actionability analysis finished.")
    print(f"Input counterfactual results: {cf_results_path}")
    print(f"Input feature changes:       {cf_changes_path}")
    print(f"Output dir:                  {output_dir.resolve()}")
    print("\nActionability summary:")
    print(pd.Series({
        "total_counterfactuals": len(cf_summary),
        "only_actionable": int(cf_summary["only_actionable"].sum()),
        "uses_non_actionable": int(cf_summary["uses_non_actionable"].sum()),
        "mean_total_changes": cf_summary["n_total_changes"].mean(),
        "mean_actionable_changes": cf_summary["n_actionable_changes"].mean(),
        "mean_non_actionable_changes": cf_summary["n_non_actionable_changes"].mean(),
    }))


if __name__ == "__main__":
    main()
