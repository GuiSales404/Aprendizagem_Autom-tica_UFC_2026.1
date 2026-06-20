#!/usr/bin/env python3
# evaluate_results.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate tuned individual models, simple average, and stacking results."
    )

    parser.add_argument("--tuning-dir", default="tuning_results")
    parser.add_argument("--stacking-dir", default="stacking_results")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


# ============================================================
# METRICS
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    values = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(metric_fn(y_true[idx], y_pred[idx]))

    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_inputs(args):
    tuning_summary_path = Path(args.tuning_dir) / "summary_all_models.csv"
    stacking_metrics_path = Path(args.stacking_dir) / "stacking_comparison_metrics.csv"
    predictions_path = Path(args.stacking_dir) / "test_predictions_stacking.csv"

    required = [
        tuning_summary_path,
        stacking_metrics_path,
        predictions_path,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    return {
        "tuning_summary": pd.read_csv(tuning_summary_path),
        "stacking_metrics": pd.read_csv(stacking_metrics_path),
        "predictions": pd.read_csv(predictions_path),
    }


# ============================================================
# PLOTS
# ============================================================

def plot_model_ranking(metrics_df, output_dir):
    plot_df = metrics_df.sort_values("rmse").copy()

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["model_name"], plot_df["rmse"])
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.title("Model comparison by test RMSE")
    plt.xticks(rotation=35, ha="right")

    for i, row in enumerate(plot_df.itertuples()):
        plt.text(i, row.rmse, f"{row.rmse:.5f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "model_ranking_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_y_true_vs_pred(pred_df, pred_col, model_label, output_dir):
    y_true = pred_df["y_true"].values
    y_pred = pred_df[pred_col].values

    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.65)
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title(f"True vs predicted — {model_label}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"true_vs_pred_{model_label}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_histogram(pred_df, pred_col, model_label, output_dir):
    residuals = pred_df["y_true"] - pred_df[pred_col]

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual: y_true - y_pred")
    plt.ylabel("Count")
    plt.title(f"Residual distribution — {model_label}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"residual_histogram_{model_label}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals_vs_pred(pred_df, pred_col, model_label, output_dir):
    residuals = pred_df["y_true"] - pred_df[pred_col]

    plt.figure(figsize=(8, 5))
    plt.scatter(pred_df[pred_col], residuals, alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted value")
    plt.ylabel("Residual: y_true - y_pred")
    plt.title(f"Residuals vs predictions — {model_label}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"residuals_vs_pred_{model_label}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_bland_altman(pred_df, pred_col, model_label, output_dir):
    y_true = pred_df["y_true"].values
    y_pred = pred_df[pred_col].values

    mean_values = (y_true + y_pred) / 2
    diff_values = y_true - y_pred

    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)

    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(8, 5))
    plt.scatter(mean_values, diff_values, alpha=0.65)
    plt.axhline(mean_diff, linestyle="-", label=f"Mean diff = {mean_diff:.4f}")
    plt.axhline(upper, linestyle="--", label=f"+1.96 SD = {upper:.4f}")
    plt.axhline(lower, linestyle="--", label=f"-1.96 SD = {lower:.4f}")
    plt.xlabel("Mean of true and predicted")
    plt.ylabel("Difference: y_true - y_pred")
    plt.title(f"Bland–Altman plot — {model_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"bland_altman_{model_label}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_absolute_error_comparison(pred_df, best_base_col, stacking_col, output_dir):
    err_base = np.abs(pred_df["y_true"] - pred_df[best_base_col])
    err_stack = np.abs(pred_df["y_true"] - pred_df[stacking_col])

    improvement = err_base - err_stack

    plt.figure(figsize=(8, 5))
    plt.hist(improvement, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Absolute error improvement: best base - stacking")
    plt.ylabel("Count")
    plt.title("Where stacking improves over the best individual model")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "stacking_error_improvement_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    comparison_df = pd.DataFrame({
        "y_true": pred_df["y_true"],
        "best_base_pred": pred_df[best_base_col],
        "stacking_pred": pred_df[stacking_col],
        "abs_error_best_base": err_base,
        "abs_error_stacking": err_stack,
        "improvement": improvement,
        "stacking_better": improvement > 0,
    })

    comparison_df.to_csv(Path(output_dir) / "sample_level_stacking_improvement.csv", index=False)

    return comparison_df


def plot_prediction_correlation(pred_df, output_dir):
    pred_cols = [c for c in pred_df.columns if c.endswith("_pred")]

    corr = pred_df[pred_cols].corr()
    corr.to_csv(Path(output_dir) / "test_prediction_correlation.csv")

    plt.figure(figsize=(8, 7))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation between test predictions")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "test_prediction_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()

    return corr


# ============================================================
# REPORT
# ============================================================

def build_metrics_from_predictions(pred_df, n_bootstrap, random_state):
    y_true = pred_df["y_true"].values

    pred_cols = [c for c in pred_df.columns if c.endswith("_pred")]

    rows = []

    for col in pred_cols:
        y_pred = pred_df[col].values
        model_name = col.replace("_pred", "")

        base_metrics = metrics_dict(y_true, y_pred)

        rmse_ci = bootstrap_ci(
            y_true,
            y_pred,
            rmse,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        mae_ci = bootstrap_ci(
            y_true,
            y_pred,
            mean_absolute_error,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        rows.append({
            "model_name": model_name,
            "rmse": base_metrics["rmse"],
            "rmse_ci_low": rmse_ci["ci_low"],
            "rmse_ci_high": rmse_ci["ci_high"],
            "mae": base_metrics["mae"],
            "mae_ci_low": mae_ci["ci_low"],
            "mae_ci_high": mae_ci["ci_high"],
            "r2": base_metrics["r2"],
        })

    return pd.DataFrame(rows).sort_values("rmse")


def write_markdown_report(metrics_df, improvement_df, corr_df, output_dir):
    best = metrics_df.iloc[0]
    best_individual = metrics_df[
        ~metrics_df["model_name"].isin(["stacking", "simple_average"])
    ].iloc[0]

    stacking_row = metrics_df[metrics_df["model_name"] == "stacking"]
    if len(stacking_row) > 0:
        stacking_row = stacking_row.iloc[0]
        stacking_gain = (
            (best_individual["rmse"] - stacking_row["rmse"]) / best_individual["rmse"]
        ) * 100
    else:
        stacking_gain = None

    pct_stacking_better = improvement_df["stacking_better"].mean() * 100

    report = []
    report.append("# Model Evaluation Report\n")
    report.append("## Overall ranking\n")
    report.append(metrics_df.to_markdown(index=False))
    report.append("\n\n## Main findings\n")
    report.append(
        f"- Best overall model by RMSE: **{best['model_name']}** "
        f"(RMSE = {best['rmse']:.5f}, MAE = {best['mae']:.5f}, R² = {best['r2']:.5f})."
    )
    report.append(
        f"- Best individual/base model: **{best_individual['model_name']}** "
        f"(RMSE = {best_individual['rmse']:.5f})."
    )

    if stacking_gain is not None:
        report.append(
            f"- Stacking changed RMSE by **{stacking_gain:.2f}%** relative to the best individual model."
        )

    report.append(
        f"- Stacking had lower absolute error than the best individual model in "
        f"**{pct_stacking_better:.1f}%** of test samples."
    )

    report.append("\n\n## Prediction correlation\n")
    report.append(corr_df.to_markdown())

    report.append("\n\n## Generated figures\n")
    for fig in sorted(Path(output_dir).glob("*.png")):
        report.append(f"- `{fig.name}`")

    Path(output_dir, "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    pred_df = inputs["predictions"].copy()

    # Normalize prediction column names generated by run_stacking.py
    rename_map = {
        "stacking_pred": "stacking_pred",
        "simple_average_pred": "simple_average_pred",
    }
    pred_df = pred_df.rename(columns=rename_map)

    metrics_df = build_metrics_from_predictions(
        pred_df,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )

    metrics_df.to_csv(output_dir / "evaluation_metrics_with_ci.csv", index=False)

    plot_model_ranking(metrics_df, output_dir)
    corr_df = plot_prediction_correlation(pred_df, output_dir)

    best_model_name = metrics_df.iloc[0]["model_name"]
    best_pred_col = f"{best_model_name}_pred"

    best_individual_name = metrics_df[
        ~metrics_df["model_name"].isin(["stacking", "simple_average"])
    ].iloc[0]["model_name"]

    best_individual_col = f"{best_individual_name}_pred"

    # Core plots for best model and stacking
    for model_name in [best_model_name, "stacking", "simple_average", best_individual_name]:
        pred_col = f"{model_name}_pred"
        if pred_col not in pred_df.columns:
            continue

        plot_y_true_vs_pred(pred_df, pred_col, model_name, output_dir)
        plot_residual_histogram(pred_df, pred_col, model_name, output_dir)
        plot_residuals_vs_pred(pred_df, pred_col, model_name, output_dir)
        plot_bland_altman(pred_df, pred_col, model_name, output_dir)

    improvement_df = plot_absolute_error_comparison(
        pred_df=pred_df,
        best_base_col=best_individual_col,
        stacking_col="stacking_pred",
        output_dir=output_dir,
    )

    write_markdown_report(
        metrics_df=metrics_df,
        improvement_df=improvement_df,
        corr_df=corr_df,
        output_dir=output_dir,
    )

    print("\nEvaluation finished.")
    print(f"Results saved to: {output_dir.resolve()}")
    print("\nRanking:")
    print(metrics_df)


if __name__ == "__main__":
    main()
