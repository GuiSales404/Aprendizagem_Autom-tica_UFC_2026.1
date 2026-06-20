#!/usr/bin/env python3
# run_stacking.py

import os
import json
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import RidgeCV, LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run regression stacking using tuned base models.")
    parser.add_argument("--splits-dir", default="../Datasets/Split")
    parser.add_argument("--tuning-dir", default="tuning_results")
    parser.add_argument("--output-dir", default="stacking_results")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "catboost", "random_forest", "xgboost", "mlp"],
    )
    parser.add_argument(
        "--meta-model",
        choices=["ridge", "linear", "elasticnet", "random_forest"],
        default="ridge",
    )
    return parser.parse_args()


def load_y(path):
    return pd.read_parquet(path).iloc[:, 0]


def load_data(splits_dir):
    splits_dir = Path(splits_dir)
    return {
        "X_train": pd.read_parquet(splits_dir / "X_train_raw.parquet"),
        "X_test": pd.read_parquet(splits_dir / "X_test_raw.parquet"),
        "y_train": load_y(splits_dir / "y_train.parquet"),
        "y_test": load_y(splits_dir / "y_test.parquet"),
    }


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_columns(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(X, scale_numeric):
    numeric_cols, categorical_cols = get_columns(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []

    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe()),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


class SafeSelectKBest(SelectKBest):
    def fit(self, X, y=None):
        if isinstance(self.k, int):
            self.k = min(self.k, X.shape[1])
        return super().fit(X, y)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_best_results(tuning_dir, model_name):
    path = Path(tuning_dir) / model_name / "best_results.json"

    if not path.exists():
        raise FileNotFoundError(f"Não encontrei {path}. Rode o tuning antes do stacking.")

    with open(path, "r") as f:
        return json.load(f)


def build_estimator(model_name, params, random_state):
    params = dict(params)
    params.pop("k_features", None)

    if model_name == "lightgbm":
        return LGBMRegressor(
            **params,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "xgboost":
        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "catboost":
        return CatBoostRegressor(
            **params,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "mlp":
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        return MLPRegressor(
            **params,
            max_iter=1000,
            early_stopping=True,
            random_state=random_state,
        )

    raise ValueError(f"Modelo não suportado: {model_name}")


def build_base_pipeline(model_name, tuning_dir, random_state, X_train):
    best = load_best_results(tuning_dir, model_name)

    params = best["best_params"]
    k_features = int(params["k_features"])

    estimator = build_estimator(
        model_name=model_name,
        params=params,
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("preprocess", make_preprocessor(
            X_train,
            scale_numeric=(model_name == "mlp"),
        )),
        ("feature_selection", SafeSelectKBest(
            score_func=mutual_info_regression,
            k=k_features,
        )),
        ("model", estimator),
    ])

    return pipeline, best


def build_meta_model(meta_model_name, random_state):
    if meta_model_name == "ridge":
        return RidgeCV(alphas=np.logspace(-6, 3, 50))

    if meta_model_name == "linear":
        return LinearRegression()

    if meta_model_name == "elasticnet":
        return ElasticNetCV(
            alphas=np.logspace(-6, 1, 40),
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5,
            random_state=random_state,
            max_iter=10000,
        )

    if meta_model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Meta-modelo não suportado: {meta_model_name}")


def generate_oof_predictions(args, data, base_model_names):
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    cv = KFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    oof_train = pd.DataFrame(index=np.arange(len(y_train)))
    test_predictions = pd.DataFrame(index=np.arange(len(y_test)))
    base_test_metrics = []
    final_base_pipelines = {}

    for model_name in base_model_names:
        print(f"\nGerando OOF para modelo base: {model_name}")

        base_pipeline_template, best_info = build_base_pipeline(
            model_name=model_name,
            tuning_dir=args.tuning_dir,
            random_state=args.random_state,
            X_train=X_train,
        )

        oof_pred = np.zeros(len(y_train), dtype=float)
        fold_metrics = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
            print(f"  Fold {fold_idx}/{args.n_splits}")

            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[valid_idx]

            fold_pipeline = clone(base_pipeline_template)
            fold_pipeline.fit(X_tr, y_tr)

            val_pred = fold_pipeline.predict(X_val)
            oof_pred[valid_idx] = val_pred

            fold_result = regression_metrics(y_val, val_pred)
            fold_result["fold"] = fold_idx
            fold_metrics.append(fold_result)

        oof_train[model_name] = oof_pred

        final_pipeline = clone(base_pipeline_template)
        final_pipeline.fit(X_train, y_train)
        final_base_pipelines[model_name] = final_pipeline

        test_pred = final_pipeline.predict(X_test)
        test_predictions[model_name] = test_pred

        base_metrics = regression_metrics(y_test, test_pred)
        base_metrics.update({
            "model_name": model_name,
            "best_cv_rmse_from_tuning": best_info.get("best_cv_rmse"),
            "n_selected_features": best_info.get("n_selected_features"),
        })

        base_test_metrics.append(base_metrics)

        pd.DataFrame(fold_metrics).to_csv(
            Path(args.output_dir) / f"{model_name}_oof_fold_metrics.csv",
            index=False,
        )

    return oof_train, test_predictions, pd.DataFrame(base_test_metrics), final_base_pipelines


def run_stacking(args):
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.splits_dir)

    y_train = data["y_train"]
    y_test = data["y_test"]

    oof_train, base_test_pred, base_metrics_df, final_base_pipelines = generate_oof_predictions(
        args=args,
        data=data,
        base_model_names=args.models,
    )

    print("\nTreinando meta-modelo...")

    meta_model = build_meta_model(args.meta_model, args.random_state)
    meta_model.fit(oof_train, y_train)

    stacked_pred = meta_model.predict(base_test_pred)
    stacked_metrics = regression_metrics(y_test, stacked_pred)
    stacked_metrics.update({
        "model_name": f"stacking_{args.meta_model}",
        "base_models": args.models,
    })

    simple_avg_pred = base_test_pred.mean(axis=1).values
    simple_avg_metrics = regression_metrics(y_test, simple_avg_pred)
    simple_avg_metrics.update({
        "model_name": "simple_average",
        "base_models": args.models,
    })

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "stacking_pred": stacked_pred,
        "simple_average_pred": simple_avg_pred,
    })

    for col in base_test_pred.columns:
        pred_df[f"{col}_pred"] = base_test_pred[col].values

    pred_df.to_csv(Path(args.output_dir) / "test_predictions_stacking.csv", index=False)

    oof_train.to_csv(Path(args.output_dir) / "oof_train_predictions.csv", index=False)
    base_test_pred.to_csv(Path(args.output_dir) / "base_test_predictions.csv", index=False)

    comparison_rows = []

    for row in base_metrics_df.to_dict(orient="records"):
        comparison_rows.append({
            "model_name": row["model_name"],
            "rmse": row["rmse"],
            "mae": row["mae"],
            "r2": row["r2"],
            "type": "base_model",
        })

    comparison_rows.append({
        "model_name": "simple_average",
        "rmse": simple_avg_metrics["rmse"],
        "mae": simple_avg_metrics["mae"],
        "r2": simple_avg_metrics["r2"],
        "type": "ensemble",
    })

    comparison_rows.append({
        "model_name": f"stacking_{args.meta_model}",
        "rmse": stacked_metrics["rmse"],
        "mae": stacked_metrics["mae"],
        "r2": stacked_metrics["r2"],
        "type": "ensemble",
    })

    comparison_df = pd.DataFrame(comparison_rows).sort_values("rmse")
    comparison_df.to_csv(Path(args.output_dir) / "stacking_comparison_metrics.csv", index=False)

    with open(Path(args.output_dir) / "stacking_results.json", "w") as f:
        json.dump(
            {
                "stacking": stacked_metrics,
                "simple_average": simple_avg_metrics,
                "base_models": base_metrics_df.to_dict(orient="records"),
                "meta_model": args.meta_model,
                "models": args.models,
                "n_splits": args.n_splits,
                "random_state": args.random_state,
            },
            f,
            indent=4,
        )

    joblib.dump(
        {
            "meta_model": meta_model,
            "base_pipelines": final_base_pipelines,
            "base_model_names": args.models,
            "meta_model_name": args.meta_model,
        },
        Path(args.output_dir) / "stacking_model.joblib",
    )

    if hasattr(meta_model, "coef_"):
        coef_df = pd.DataFrame({
            "base_model": oof_train.columns,
            "meta_coefficient": np.ravel(meta_model.coef_),
        }).sort_values("meta_coefficient", ascending=False)

        coef_df.to_csv(Path(args.output_dir) / "meta_model_coefficients.csv", index=False)

    corr_df = oof_train.corr()
    corr_df.to_csv(Path(args.output_dir) / "oof_prediction_correlation.csv")

    plot_comparison(comparison_df, args.output_dir)
    plot_prediction_correlation(corr_df, args.output_dir)
    plot_meta_coefficients(args.output_dir)

    print("\nResumo final:")
    print(comparison_df)
    print(f"\nResultados salvos em: {args.output_dir}")


def plot_comparison(comparison_df, output_dir):
    plot_df = comparison_df.sort_values("rmse")

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["model_name"], plot_df["rmse"])
    plt.ylabel("Test RMSE")
    plt.xlabel("Modelo")
    plt.title("Comparação de modelos base, média simples e stacking")
    plt.xticks(rotation=35, ha="right")

    for i, row in enumerate(plot_df.itertuples()):
        plt.text(i, row.rmse, f"{row.rmse:.5f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "stacking_comparison_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_correlation(corr_df, output_dir):
    plt.figure(figsize=(7, 6))
    plt.imshow(corr_df.values, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title("Correlação entre predições OOF dos modelos base")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "oof_prediction_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_meta_coefficients(output_dir):
    coef_path = Path(output_dir) / "meta_model_coefficients.csv"

    if not coef_path.exists():
        return

    coef_df = pd.read_csv(coef_path).sort_values("meta_coefficient", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(coef_df["base_model"], coef_df["meta_coefficient"])
    plt.ylabel("Coeficiente no meta-modelo")
    plt.xlabel("Modelo base")
    plt.title("Peso aprendido pelo meta-modelo")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "meta_model_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    run_stacking(args)


if __name__ == "__main__":
    main()