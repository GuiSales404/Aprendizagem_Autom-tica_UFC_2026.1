# train_individual_models.py

import os
import json
import argparse
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="../Datasets/Split")
    parser.add_argument("--output-dir", default="tuning_results")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "catboost", "random_forest", "xgboost", "mlp"],
    )
    return parser.parse_args()


def load_y(path):
    return pd.read_parquet(path).iloc[:, 0]


def load_data(splits_dir):
    return {
        "X_train": pd.read_parquet(f"{splits_dir}/X_train_raw.parquet"),
        "X_test": pd.read_parquet(f"{splits_dir}/X_test_raw.parquet"),
        "y_train": load_y(f"{splits_dir}/y_train.parquet"),
        "y_test": load_y(f"{splits_dir}/y_test.parquet"),
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

    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])

    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

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
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_model(trial, model_name, random_state):
    if model_name == "lightgbm":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "catboost":
        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=trial.suggest_categorical(
                "hidden_layer_sizes",
                [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            ),
            activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            max_iter=1000,
            early_stopping=True,
            random_state=random_state,
        )

    raise ValueError(f"Modelo não suportado: {model_name}")


def make_pipeline(model, k_features, X, model_name):
    scale_numeric = model_name == "mlp"

    return Pipeline([
        ("preprocess", make_preprocessor(X, scale_numeric=scale_numeric)),
        ("feature_selection", SafeSelectKBest(score_func=mutual_info_regression, k=k_features)),
        ("model", model),
    ])


def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        n = pipeline.named_steps["feature_selection"].scores_.shape[0]
        return [f"feature_{i}" for i in range(n)]


def plot_top10(trials_df, model_dir, model_name):
    top10 = trials_df.sort_values("value").head(10).copy()
    top10["label"] = top10["number"].apply(lambda x: f"Trial {int(x)}")

    plt.figure(figsize=(12, 5))
    plt.bar(top10["label"], top10["value"])
    plt.ylabel("CV RMSE")
    plt.xlabel("Combinação")
    plt.title(f"Top 10 combinações — {model_name}")

    for i, row in enumerate(top10.itertuples()):
        plt.text(i, row.value, f"{row.value:.5f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{model_dir}/top10_trials_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()


def tune_model(model_name, X_train, X_test, y_train, y_test, args):
    model_dir = f"{args.output_dir}/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # Como OHE aumenta o número de features, usamos um limite seguro.
    max_k = 100
    min_k = 5

    cv = KFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    def objective(trial):
        k_features = trial.suggest_int("k_features", min_k, max_k)

        model = get_model(trial, model_name, args.random_state)

        pipeline = make_pipeline(
            model=model,
            k_features=k_features,
            X=X_train,
            model_name=model_name,
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )

        trial.set_user_attr("rmse_cv_std", float(scores.std()))
        return float(-scores.mean())

    study = optuna.create_study(direction="minimize", study_name=f"{model_name}_tuning")
    study.optimize(objective, n_trials=args.n_trials)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{model_dir}/all_trials.csv", index=False)

    best_params = study.best_trial.params
    best_k = best_params["k_features"]

    best_model = get_model(
        optuna.trial.FixedTrial(best_params),
        model_name,
        args.random_state,
    )

    best_pipeline = make_pipeline(
        model=best_model,
        k_features=best_k,
        X=X_train,
        model_name=model_name,
    )

    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)

    test_rmse = rmse(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    selector = best_pipeline.named_steps["feature_selection"]
    selected_mask = selector.get_support()

    feature_names = get_feature_names(best_pipeline)
    selected_features = np.array(feature_names)[selected_mask].tolist()

    result = {
        "model_name": model_name,
        "best_cv_rmse": float(study.best_value),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "best_params": best_params,
        "n_selected_features": len(selected_features),
        "selected_features": selected_features,
    }

    with open(f"{model_dir}/best_results.json", "w") as f:
        json.dump(result, f, indent=4)

    joblib.dump(best_pipeline, f"{model_dir}/best_pipeline.joblib")

    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    }).to_csv(f"{model_dir}/test_predictions.csv", index=False)

    pd.DataFrame({
        "feature": feature_names,
        "selection_score": selector.scores_,
        "selected": selected_mask,
    }).sort_values("selection_score", ascending=False).to_csv(
        f"{model_dir}/feature_selection_scores.csv",
        index=False,
    )

    plot_top10(trials_df, model_dir, model_name)

    return result


def plot_summary(summary_df, output_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["model_name"], summary_df["test_rmse"])
    plt.ylabel("Test RMSE")
    plt.xlabel("Modelo")
    plt.title("Comparação final dos modelos tunados")

    for i, row in enumerate(summary_df.itertuples()):
        plt.text(i, row.test_rmse, f"{row.test_rmse:.5f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_test_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.splits_dir)

    all_results = []

    for model_name in args.models:
        print(f"\nRodando tuning: {model_name}")

        result = tune_model(
            model_name=model_name,
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=data["y_train"],
            y_test=data["y_test"],
            args=args,
        )

        all_results.append(result)

    summary_df = pd.DataFrame([
        {
            "model_name": r["model_name"],
            "best_cv_rmse": r["best_cv_rmse"],
            "test_rmse": r["test_rmse"],
            "test_mae": r["test_mae"],
            "test_r2": r["test_r2"],
            "n_selected_features": r["n_selected_features"],
        }
        for r in all_results
    ]).sort_values("test_rmse")

    summary_df.to_csv(f"{args.output_dir}/summary_all_models.csv", index=False)
    plot_summary(summary_df, args.output_dir)

    print("\nResumo final:")
    print(summary_df)


if __name__ == "__main__":
    main()