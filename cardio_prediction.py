from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = "cardio"
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc"]
NUMERIC_FEATURES = ["age_years", "height", "weight", "ap_hi", "ap_lo", "smoke", "alco", "active"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = {
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", TARGET,
}


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    try:
        raw = pd.read_csv(csv_path, sep=None, engine="python")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read dataset at {csv_path}: {exc}") from exc

    raw.columns = [str(column).strip().lower() for column in raw.columns]
    missing_columns = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    initial_rows = len(raw)
    data = raw.copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.loc[data[TARGET].isin([0, 1])].copy()
    invalid_target_rows = initial_rows - len(data)

    x = clean_features(data)
    y = data[TARGET].astype(int).copy()
    summary = {
        "input_rows": int(initial_rows),
        "rows_removed_invalid_target": int(invalid_target_rows),
        "rows_used": int(len(data)),
        "features_used": int(len(FEATURES)),
    }
    return x, y, summary


def clean_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in ["age", "height", "weight", "ap_hi", "ap_lo", "smoke", "alco", "active"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["age_years"] = data["age"] / 365.25
    data.loc[~data["age_years"].between(18, 110), "age_years"] = np.nan
    data.loc[~data["height"].between(100, 250), "height"] = np.nan
    data.loc[~data["weight"].between(30, 300), "weight"] = np.nan

    valid_bp = (
        data["ap_hi"].between(60, 250)
        & data["ap_lo"].between(30, 200)
        & (data["ap_hi"] >= data["ap_lo"])
    )
    data.loc[~valid_bp, ["ap_hi", "ap_lo"]] = np.nan

    for column in ["smoke", "alco", "active"]:
        data.loc[~data[column].isin([0, 1]), column] = np.nan
    for column in CATEGORICAL_FEATURES:
        data.loc[~data[column].isin([1, 2, 3] if column != "gender" else [1, 2]), column] = np.nan

    return data[FEATURES].copy()


def load_patient_records(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    try:
        raw = pd.read_csv(csv_path, sep=None, engine="python")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read patient records at {csv_path}: {exc}") from exc

    raw.columns = [str(column).strip().lower() for column in raw.columns]
    missing_columns = sorted((REQUIRED_COLUMNS - {TARGET}) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Patient records are missing required columns: {', '.join(missing_columns)}")
    row_ids = raw["id"].copy() if "id" in raw.columns else pd.Series(raw.index, name="row")
    return clean_features(raw), row_ids


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, Any]:
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=42
        ),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=150,
            n_iter_no_change=10,
            random_state=42,
        ),
    }
    return {
        name: Pipeline([("preprocessor", make_preprocessor()), ("classifier", classifier)])
        for name, classifier in classifiers.items()
    }


def evaluate_models(
    models: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    comparison: list[dict[str, Any]] = []
    test_metrics: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    scoring = {"accuracy": "accuracy", "f1": "f1"}

    for name, model in models.items():
        print(f"Cross-validating and fitting {name}...", flush=True)
        cv_result = cross_validate(
            model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
        )
        comparison.append(
            {
                "model": name,
                "cv_accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
                "cv_accuracy_std": float(np.std(cv_result["test_accuracy"])),
                "cv_f1_mean": float(np.mean(cv_result["test_f1"])),
                "cv_f1_std": float(np.std(cv_result["test_f1"])),
            }
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        fitted_models[name] = model
        test_metrics.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
                "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                ).ravel().tolist(),
            }
        )

    return pd.DataFrame(comparison), pd.DataFrame(test_metrics), fitted_models


def save_confusion_matrices(
    models: dict[str, Any], x_test: pd.DataFrame, y_test: pd.Series, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (name, model) in zip(axes.flat, models.items()):
        display = ConfusionMatrixDisplay.from_estimator(
            model, x_test, y_test, display_labels=["No disease", "Disease"],
            cmap="Blues", colorbar=False, ax=axis,
        )
        display.ax_.set_title(name)
    fig.suptitle("Held-out test set confusion matrices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(
    best_model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    sample_size = min(2000, len(x_test))
    sample = x_test.sample(n=sample_size, random_state=seed)
    labels = y_test.loc[sample.index]
    result = permutation_importance(
        best_model,
        sample,
        labels,
        scoring="f1",
        n_repeats=3,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean_f1_decrease": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean_f1_decrease", ascending=False, ignore_index=True)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    plotted = importance.sort_values("importance_mean_f1_decrease")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        plotted["feature"],
        plotted["importance_mean_f1_decrease"],
        xerr=plotted["importance_std"],
        color="#3478a8",
    )
    axis.set_xlabel("Decrease in F1 when feature is permuted")
    axis.set_title("Permutation feature importance (best model, test sample)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return importance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare four cardiovascular disease classifiers."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/cardio_train.csv"),
        help="Path to cardio_train.csv (default: data/cardio_train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for metrics, figures, and the selected model.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds (default: 5)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--predict-input",
        type=Path,
        help="Optional unlabeled patient CSV to classify after training.",
    )
    args = parser.parse_args(argv)

    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    try:
        x, y, data_summary = load_and_clean_data(args.data)
        if y.value_counts().min() < args.cv_folds:
            raise ValueError("Each target class must have at least as many rows as --cv-folds.")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)

        cv_results, test_results, fitted = evaluate_models(
            build_models(), x_train, y_train, x_test, y_test, args.cv_folds, args.seed
        )
        cv_results = cv_results.sort_values("cv_f1_mean", ascending=False, ignore_index=True)
        test_results.to_csv(args.output_dir / "test_metrics.csv", index=False)
        cv_results.to_csv(args.output_dir / "cross_validation_metrics.csv", index=False)
        save_confusion_matrices(
            fitted, x_test, y_test, args.output_dir / "confusion_matrices.png"
        )

        best_name = str(cv_results.iloc[0]["model"])
        best_model = fitted[best_name]
        importance = save_feature_importance(
            best_model, x_test, y_test, args.output_dir, args.seed
        )
        joblib.dump(best_model, args.output_dir / "best_model.joblib")
        if args.predict_input:
            patients, row_ids = load_patient_records(args.predict_input)
            predictions = best_model.predict(patients)
            prediction_output = pd.DataFrame(
                {
                    row_ids.name or "row": row_ids.to_numpy(),
                    "predicted_cardio": predictions.astype(int),
                }
            )
            if hasattr(best_model, "predict_proba"):
                prediction_output["estimated_probability_cardio_1"] = best_model.predict_proba(
                    patients
                )[:, 1]
            prediction_output.to_csv(args.output_dir / "predictions.csv", index=False)

        summary = {
            "dataset": data_summary,
            "split": {
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "test_fraction": args.test_size,
                "stratified": True,
            },
            "cross_validation_folds": args.cv_folds,
            "selected_model": best_name,
            "selection_metric": "mean cross-validation F1-score",
            "best_model_test_metrics": test_results.set_index("model").loc[best_name].to_dict(),
            "top_features": importance.head(5)["feature"].tolist(),
            "clinical_notice": (
                "Educational research output only; not a diagnostic tool or substitute "
                "for assessment by a qualified healthcare professional."
            ),
        }
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        print("\nCross-validation comparison (ranked by mean F1):")
        print(cv_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print("\nHeld-out test metrics:")
        print(test_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"\nSelected model: {best_name}")
        if args.predict_input:
            print(f"Predictions saved to: {(args.output_dir / 'predictions.csv').resolve()}")
        print(f"Results saved to: {args.output_dir.resolve()}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
