#!/usr/bin/env python3
# explain_best_model.py

import os
import json
import argparse
import warnings
from pathlib import Path
import shap
from dice_ml import Dice, Data, Model
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest

warnings.filterwarnings("ignore")


class SafeSelectKBest(SelectKBest):
    def fit(self, X, y=None):
        if isinstance(self.k, int):
            self.k = min(self.k, X.shape[1])
        return super().fit(X, y)

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select the best tuned model and generate model-agnostic explanations "
            "using SHAP and counterfactual explanations using DiCE."
        )
    )

    parser.add_argument("--splits-dir", default="../Datasets/Split")
    parser.add_argument("--tuning-dir", default="tuning_results")
    parser.add_argument("--stacking-dir", default="stacking_results")
    parser.add_argument("--output-dir", default="explainability_results")

    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Model to explain. If omitted, the best individual model is selected "
            "from stacking_comparison_metrics.csv or summary_all_models.csv."
        ),
    )

    parser.add_argument(
        "--explain-stacking",
        action="store_true",
        help=(
            "Explain the saved stacking model instead of an individual model. "
            "SHAP works, but counterfactuals are skipped by default for stacking."
        ),
    )

    parser.add_argument("--n-background", type=int, default=80)
    parser.add_argument("--n-explain", type=int, default=40)
    parser.add_argument("--n-counterfactuals", type=int, default=3)
    parser.add_argument("--n-cf-instances", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--desired-low",
        type=float,
        default=None,
        help="Lower bound for desired prediction range in counterfactual explanations.",
    )
    parser.add_argument(
        "--desired-high",
        type=float,
        default=None,
        help="Upper bound for desired prediction range in counterfactual explanations.",
    )

    parser.add_argument(
        "--cf-mode",
        choices=["lower", "higher", "range"],
        default="lower",
        help=(
            "Counterfactual target strategy when desired-low/high are not provided. "
            "'lower' searches for predictions below the original value; "
            "'higher' searches for predictions above it; "
            "'range' uses the empirical 25th-75th percentile range of y_train."
        ),
    )

    return parser.parse_args()


# ============================================================
# Data loading
# ============================================================

def load_y(path):
    return pd.read_parquet(path).iloc[:, 0]


def load_data(splits_dir):
    splits_dir = Path(splits_dir)

    required = [
        splits_dir / "X_train_raw.parquet",
        splits_dir / "X_test_raw.parquet",
        splits_dir / "y_train.parquet",
        splits_dir / "y_test.parquet",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {path}")

    return {
        "X_train": pd.read_parquet(splits_dir / "X_train_raw.parquet"),
        "X_test": pd.read_parquet(splits_dir / "X_test_raw.parquet"),
        "y_train": load_y(splits_dir / "y_train.parquet"),
        "y_test": load_y(splits_dir / "y_test.parquet"),
    }


# ============================================================
# Model selection / loading
# ============================================================

def select_best_individual_model(tuning_dir, stacking_dir):
    stacking_metrics_path = Path(stacking_dir) / "stacking_comparison_metrics.csv"

    if stacking_metrics_path.exists():
        df = pd.read_csv(stacking_metrics_path)

        if "type" in df.columns:
            df = df[df["type"] == "base_model"].copy()

        if len(df) > 0 and "rmse" in df.columns:
            return df.sort_values("rmse").iloc[0]["model_name"]

    summary_path = Path(tuning_dir) / "summary_all_models.csv"

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        return df.sort_values("test_rmse").iloc[0]["model_name"]

    raise FileNotFoundError(
        "Não consegui encontrar ranking em stacking_comparison_metrics.csv "
        "nem summary_all_models.csv."
    )


def load_individual_pipeline(tuning_dir, model_name):
    model_path = Path(tuning_dir) / model_name / "best_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo salvo não encontrado: {model_path}")

    return joblib.load(model_path)


def load_stacking_model(stacking_dir):
    model_path = Path(stacking_dir) / "stacking_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Stacking salvo não encontrado: {model_path}")

    return joblib.load(model_path)


def make_predict_fn_individual(model, feature_names):
    def predict_fn(X):
        X_df = pd.DataFrame(X, columns=feature_names)
        return np.asarray(model.predict(X_df)).reshape(-1)

    return predict_fn


def make_predict_fn_stacking(artifact, feature_names):
    meta_model = artifact["meta_model"]
    base_pipelines = artifact["base_pipelines"]
    base_model_names = artifact["base_model_names"]

    def predict_fn(X):
        X_df = pd.DataFrame(X, columns=feature_names)

        base_preds = pd.DataFrame({
            model_name: base_pipelines[model_name].predict(X_df)
            for model_name in base_model_names
        })

        return np.asarray(meta_model.predict(base_preds)).reshape(-1)

    return predict_fn


# ============================================================
# Metrics
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(predict_fn, X_test, y_test):
    y_pred = predict_fn(X_test)

    return {
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }, y_pred


# ============================================================
# SHAP
# ============================================================

def sample_df(df, n, random_state):
    if len(df) <= n:
        return df.copy()

    return df.sample(n=n, random_state=random_state).copy()


def run_shap_analysis(
    predict_fn,
    X_train,
    X_test,
    output_dir,
    n_background,
    n_explain,
    random_state,
):

    shap_dir = Path(output_dir) / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    X_background = sample_df(X_train, n_background, random_state)
    X_explain = sample_df(X_test, n_explain, random_state)

    def predict_df(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=X_train.columns)
        return predict_fn(X)

    explainer = shap.KernelExplainer(
        predict_df,
        X_background,
    )

    shap_values = explainer.shap_values(
        X_explain,
        nsamples=100,
    )

    shap_values = np.asarray(shap_values)

    np.save(shap_dir / "shap_values.npy", shap_values)
    X_explain.to_csv(shap_dir / "shap_explained_instances.csv", index=False)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(shap_dir / "shap_feature_importance.csv", index=False)

    # Bar plot manual para evitar incompatibilidades com Explanation object
    top20 = importance_df.head(20).sort_values("mean_abs_shap")

    plt.figure(figsize=(8, 7))
    plt.barh(top20["feature"], top20["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("SHAP feature importance — top 20")
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_bar_top20.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Summary beeswarm compatível com KernelExplainer
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_beeswarm_top20.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "importance_path": str(shap_dir / "shap_feature_importance.csv"),
        "bar_plot": str(shap_dir / "shap_bar_top20.png"),
        "beeswarm_plot": str(shap_dir / "shap_beeswarm_top20.png"),
    }

# ============================================================
# Counterfactual explanations with DiCE
# ============================================================

def infer_feature_types(X):
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    continuous_features = [
        c for c in X.columns
        if c not in categorical_features
    ]

    return continuous_features, categorical_features


def desired_range_for_instance(
    original_pred,
    y_train,
    args,
):
    if args.desired_low is not None and args.desired_high is not None:
        return [args.desired_low, args.desired_high]

    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_std = float(np.std(y_train))

    if args.cf_mode == "range":
        return [
            float(np.percentile(y_train, 25)),
            float(np.percentile(y_train, 75)),
        ]

    if args.cf_mode == "lower":
        high = float(original_pred - 0.25 * y_std)
        low = y_min
        if high <= low:
            high = float(original_pred)
        return [low, high]

    if args.cf_mode == "higher":
        low = float(original_pred + 0.25 * y_std)
        high = y_max
        if low >= high:
            low = float(original_pred)
        return [low, high]

    raise ValueError(f"cf_mode inválido: {args.cf_mode}")


def run_counterfactual_analysis(
    model,
    X_train,
    X_test,
    y_train,
    predict_fn,
    output_dir,
    args,
):
    cf_dir = Path(output_dir) / "counterfactuals"
    cf_dir.mkdir(parents=True, exist_ok=True)

    continuous_features, categorical_features = infer_feature_types(X_train)

    target_col = "__target__"

    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    dice_data = Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name=target_col,
    )

    dice_model = Model(
        model=model,
        backend="sklearn",
        model_type="regressor",
    )

    explainer = Dice(
        dice_data,
        dice_model,
        method="random",
    )

    X_candidates = X_test.copy()
    y_pred_test = predict_fn(X_candidates)

    # Select instances with higher predicted values by default for "lower" mode,
    # or lower predicted values for "higher" mode.
    if args.cf_mode == "higher":
        selected_idx = np.argsort(y_pred_test)[:args.n_cf_instances]
    else:
        selected_idx = np.argsort(y_pred_test)[-args.n_cf_instances:]

    all_rows = []
    all_diffs = []

    for rank, idx in enumerate(selected_idx, start=1):
        query = X_candidates.iloc[[idx]].copy()
        original_pred = float(y_pred_test[idx])

        desired_range = desired_range_for_instance(
            original_pred=original_pred,
            y_train=y_train,
            args=args,
        )

        print(
            f"Gerando CF para instância {idx} "
            f"(pred={original_pred:.5f}, desired_range={desired_range})"
        )

        try:
            cf = explainer.generate_counterfactuals(
                query,
                total_CFs=args.n_counterfactuals,
                desired_range=desired_range,
            )

            cf_df = cf.cf_examples_list[0].final_cfs_df.copy()

            if cf_df is None or len(cf_df) == 0:
                continue

            if target_col in cf_df.columns:
                cf_df = cf_df.drop(columns=[target_col])

            cf_preds = predict_fn(cf_df[X_train.columns])

            original_row = query.iloc[0]

            for cf_id, (_, cf_row) in enumerate(cf_df.iterrows(), start=1):
                row = {
                    "instance_rank": rank,
                    "original_index": int(idx),
                    "counterfactual_id": cf_id,
                    "original_prediction": original_pred,
                    "counterfactual_prediction": float(cf_preds[cf_id - 1]),
                    "desired_low": desired_range[0],
                    "desired_high": desired_range[1],
                }

                for col in X_train.columns:
                    row[f"original__{col}"] = original_row[col]
                    row[f"counterfactual__{col}"] = cf_row[col]

                    if original_row[col] != cf_row[col]:
                        all_diffs.append({
                            "instance_rank": rank,
                            "original_index": int(idx),
                            "counterfactual_id": cf_id,
                            "feature": col,
                            "original_value": original_row[col],
                            "counterfactual_value": cf_row[col],
                        })

                all_rows.append(row)

        except Exception as exc:
            all_rows.append({
                "instance_rank": rank,
                "original_index": int(idx),
                "counterfactual_id": None,
                "original_prediction": original_pred,
                "counterfactual_prediction": None,
                "desired_low": desired_range[0],
                "desired_high": desired_range[1],
                "error": str(exc),
            })

    cf_results = pd.DataFrame(all_rows)
    cf_diffs = pd.DataFrame(all_diffs)

    cf_results.to_csv(cf_dir / "counterfactual_results.csv", index=False)
    cf_diffs.to_csv(cf_dir / "counterfactual_feature_changes.csv", index=False)

    if len(cf_diffs) > 0:
        change_counts = (
            cf_diffs["feature"]
            .value_counts()
            .reset_index()
        )
        change_counts.columns = ["feature", "n_changes"]
        change_counts.to_csv(cf_dir / "counterfactual_change_counts.csv", index=False)

        plt.figure(figsize=(10, 5))
        plt.bar(change_counts["feature"].head(20), change_counts["n_changes"].head(20))
        plt.ylabel("Number of changes")
        plt.xlabel("Feature")
        plt.title("Most frequently changed features in counterfactuals")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(cf_dir / "counterfactual_change_counts_top20.png", dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "counterfactual_results": str(cf_dir / "counterfactual_results.csv"),
        "counterfactual_feature_changes": str(cf_dir / "counterfactual_feature_changes.csv"),
    }


# ============================================================
# Report
# ============================================================

def write_report(output_dir, model_name, metrics, shap_info, cf_info, explain_stacking):
    report = []
    report.append("# Explainability Report\n")
    report.append(f"## Explained model\n")
    report.append(f"- Model: **{model_name}**")
    report.append(f"- Explained stacking: **{explain_stacking}**")
    report.append("\n## Test performance\n")
    report.append(f"- RMSE: **{metrics['rmse']:.6f}**")
    report.append(f"- MAE: **{metrics['mae']:.6f}**")
    report.append(f"- R²: **{metrics['r2']:.6f}**")

    report.append("\n## SHAP outputs\n")
    for key, value in shap_info.items():
        report.append(f"- {key}: `{value}`")

    report.append("\n## Counterfactual outputs\n")
    if cf_info:
        for key, value in cf_info.items():
            report.append(f"- {key}: `{value}`")
    else:
        report.append("- Counterfactuals were skipped.")

    Path(output_dir, "explainability_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.splits_dir)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    feature_names = X_train.columns.tolist()

    if args.explain_stacking:
        model_name = "stacking"
        artifact = load_stacking_model(args.stacking_dir)
        predict_fn = make_predict_fn_stacking(artifact, feature_names)
        model_for_counterfactuals = None
    else:
        model_name = args.model_name or select_best_individual_model(
            tuning_dir=args.tuning_dir,
            stacking_dir=args.stacking_dir,
        )
        model = load_individual_pipeline(args.tuning_dir, model_name)
        predict_fn = make_predict_fn_individual(model, feature_names)
        model_for_counterfactuals = model

    metrics, y_pred = evaluate_model(
        predict_fn=predict_fn,
        X_test=X_test,
        y_test=y_test,
    )

    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    }).to_csv(output_dir / "explained_model_predictions.csv", index=False)

    with open(output_dir / "explained_model_metrics.json", "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "metrics": metrics,
                "explain_stacking": args.explain_stacking,
            },
            f,
            indent=4,
        )

    print(f"\nModelo selecionado para explicabilidade: {model_name}")
    print(metrics)

    print("\nRodando SHAP...")
    shap_info = run_shap_analysis(
        predict_fn=predict_fn,
        X_train=X_train,
        X_test=X_test,
        output_dir=output_dir,
        n_background=args.n_background,
        n_explain=args.n_explain,
        random_state=args.random_state,
    )

    cf_info = None

    if not args.explain_stacking:
        print("\nRodando explicações contrafactuais com DiCE...")
        cf_info = run_counterfactual_analysis(
            model=model_for_counterfactuals,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            predict_fn=predict_fn,
            output_dir=output_dir,
            args=args,
        )
    else:
        print("\nCounterfactuals skipped for stacking model.")

    write_report(
        output_dir=output_dir,
        model_name=model_name,
        metrics=metrics,
        shap_info=shap_info,
        cf_info=cf_info,
        explain_stacking=args.explain_stacking,
    )

    print(f"\nAnálise de explicabilidade finalizada.")
    print(f"Resultados salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
