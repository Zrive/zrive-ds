"""
MVP pipeline for push notification purchase prediction.

Pipeline: load data -> validate -> preprocess -> train model -> save to disk.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, TargetEncoder

logger = logging.getLogger(__name__)

# Constants
# Schema and validation
CRITICAL_COLUMNS = ["order_id", "user_id", "outcome", "product_type"]
REQUIRED_COLUMNS = [
    "variant_id",
    "product_type",
    "order_id",
    "user_id",
    "created_at",
    "order_date",
    "user_order_seq",
    "outcome",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
    "normalised_price",
    "discount_pct",
    "vendor",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
]
# Preprocessing (dropped for multicollinearity)
DROP_COLUMNS = [
    "people_ex_baby",
    "std_days_to_buy_variant_id",
    "std_days_to_buy_product_type",
]
# Features: exclude ids, target, raw categoricals (we use _enc versions)
EXCLUDE_FEATURE_COLS = [
    "variant_id",
    "product_type",
    "order_id",
    "user_id",
    "created_at",
    "order_date",
    "outcome",
    "vendor",
]
# Dtype conversions
BOOLEAN_COLUMNS = [
    "outcome",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
]
INTEGER_COLUMNS = [
    "days_since_purchase_variant_id",
    "days_since_purchase_product_type",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
]
RANDOM_STATE = 42
# Model config (from model_decision_analysis.ipynb). Final model: LR_l1_C1.0.
CONFIG = {"penalty": "l1", "C": 1.0, "solver": "saga", "max_iter": 1000}
# Default pipeline dir for inference (matches save_pipeline output)
DEFAULT_PIPELINE_DIR = (
    Path(__file__).resolve().parent / "models" / "purchase_model_mvp_20260307"
)


# Training pipeline


def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV data from the given path."""
    df = pd.read_csv(path)
    return df


def validate_data(
    df: pd.DataFrame,
    on_null: Literal["raise", "drop"] = "raise",
    null_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Validate data schema and critical columns.

    Args:
        df: Input DataFrame.
        on_null: How to handle nulls in critical columns.
            - "raise": Fail immediately (default, for dev/CI).
            - "drop": Drop rows with nulls, log warning. If null fraction exceeds
              null_threshold, raise anyway (likely data pipeline issue).
        null_threshold: Max fraction of rows allowed with nulls when on_null="drop".
            Above this, always raise. Default 0.10 (10%).

    Returns:
        Validated DataFrame (possibly with rows dropped if on_null="drop").
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[CRITICAL_COLUMNS].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) == 0:
        return df

    n_total = len(df)
    rows_with_any_null = df[CRITICAL_COLUMNS].isna().any(axis=1).sum()
    null_frac = rows_with_any_null / n_total if n_total > 0 else 0.0

    if on_null == "raise":
        raise ValueError(
            f"Critical columns have nulls: {cols_with_nulls.to_dict()}. "
            f"Use on_null='drop' to remove rows with nulls."
        )

    if null_frac > null_threshold:
        raise ValueError(
            f"Too many nulls ({null_frac:.1%} of rows). "
            f"Nulls by column: {cols_with_nulls.to_dict()}. "
            f"Threshold: {null_threshold:.1%}."
        )

    df_clean = df.dropna(subset=CRITICAL_COLUMNS)
    n_dropped = n_total - len(df_clean)
    drop_frac = n_dropped / n_total if n_total > 0 else 0.0
    logger.warning(
        "Dropped %d rows (%s) with nulls in critical columns: %s",
        n_dropped,
        f"{drop_frac:.1%}",
        cols_with_nulls.to_dict(),
    )
    return df_clean


def preprocess(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    Preprocess data: filter, drop, cap, split, encode, scale.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, pipeline_artifacts).
    pipeline_artifacts contains: scaler, encoder, feature_cols.
    """
    df = df.copy()

    # Dtype conversions
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    # Cap discount_pct at 1
    df["discount_pct"] = df["discount_pct"].apply(lambda x: 1 if x > 1 else x)

    # Drop columns with high multicollinearity (VIF > 10).
    # people_ex_baby = count_adults + count_children; std_* are highly correlated with avg_*.
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Filter orders with at least 5 purchased items
    purchased_per_order = df.groupby("order_id")["outcome"].sum()
    valid_order_ids = purchased_per_order[purchased_per_order >= 5].index
    df = df[df["order_id"].isin(valid_order_ids)]

    # Split by user_id (to avoid data leakage)
    unique_users = df["user_id"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_users)
    n = len(unique_users)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_users = unique_users[:n_train]
    val_users = unique_users[n_train : n_train + n_val]
    test_users = unique_users[n_train + n_val :]

    df_train = df[df["user_id"].isin(train_users)].copy()
    df_val = df[df["user_id"].isin(val_users)].copy()
    df_test = df[df["user_id"].isin(test_users)].copy()

    # Target encode product_type and vendor (fit on train only to avoid leakage)
    encoder_product = TargetEncoder(
        categories="auto",
        target_type="continuous",
        smooth="auto",
        cv=5,
        random_state=RANDOM_STATE,
    )
    encoder_vendor = TargetEncoder(
        categories="auto",
        target_type="continuous",
        smooth="auto",
        cv=5,
        random_state=RANDOM_STATE,
    )
    df_train["product_type_enc"] = encoder_product.fit_transform(
        df_train[["product_type"]], df_train["outcome"]
    )
    df_val["product_type_enc"] = encoder_product.transform(df_val[["product_type"]])
    df_test["product_type_enc"] = encoder_product.transform(df_test[["product_type"]])

    df_train["vendor_enc"] = encoder_vendor.fit_transform(
        df_train[["vendor"]], df_train["outcome"]
    )
    df_val["vendor_enc"] = encoder_vendor.transform(df_val[["vendor"]])
    df_test["vendor_enc"] = encoder_vendor.transform(df_test[["vendor"]])

    # Features (exclude ids, target, raw categoricals; we use _enc versions)
    exclude_cols = [c for c in EXCLUDE_FEATURE_COLS if c in df_train.columns]
    feature_cols = [
        c
        for c in df_train.columns
        if c not in exclude_cols and df_train[c].dtype in ["int64", "float64"]
    ]

    X_train = df_train[feature_cols].values
    y_train = df_train["outcome"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["outcome"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["outcome"].values

    # Scale (fit on train only to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    artifacts = {
        "scaler": scaler,
        "encoder_product": encoder_product,
        "encoder_vendor": encoder_vendor,
        "feature_cols": feature_cols,
    }

    return (
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        artifacts,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[LogisticRegression, dict]:
    """
    Train the logistic regression model with CONFIG.
    Returns (model, metrics).
    """
    model = LogisticRegression(random_state=RANDOM_STATE, **CONFIG)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = {
        "validation_precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "validation_recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "validation_f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "validation_accuracy": float(accuracy_score(y_val, y_pred)),
    }
    return model, metrics


def save_pipeline(
    model: LogisticRegression,
    scaler: StandardScaler,
    encoder_product: TargetEncoder,
    encoder_vendor: TargetEncoder,
    metadata: dict,
    output_dir: str | Path,
) -> Path:
    """Save model, scaler, encoders and metadata to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "encoder_product.pkl", "wb") as f:
        pickle.dump(encoder_product, f)
    with open(output_dir / "encoder_vendor.pkl", "wb") as f:
        pickle.dump(encoder_vendor, f)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


# Orchestration


def main(
    data_path: str | Path | None = None,
    output_base: str | Path | None = None,
    on_null: Literal["raise", "drop"] = "raise",
) -> Path:
    """
    Run full pipeline: load, validate, preprocess, train, save.
    Returns path to the saved pipeline directory.

    Args:
        on_null: "raise" (default) to fail on nulls, "drop" to remove rows (production).
    """
    script_dir = Path(__file__).resolve().parent
    if data_path is None:
        data_path = script_dir / "module_3_datasets" / "feature_frame.csv"
    if output_base is None:
        output_base = script_dir / "models"

    output_dir = Path(output_base) / "purchase_model_mvp_20260307"

    df = load_data(data_path)
    df = validate_data(df, on_null=on_null)

    # preprocess() returns (X_train, y_train, X_val, y_val, X_test, y_test, artifacts)
    # artifacts = {scaler, encoder_product, encoder_vendor, feature_cols}
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        artifacts,
    ) = preprocess(df)

    model, metrics = train_model(X_train, y_train, X_val, y_val)

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "feature_cols": artifacts["feature_cols"],
        "config": CONFIG,
        "validation_precision": metrics["validation_precision"],
        "threshold_note": "F0.5 recommended for inference (favors precision)",
    }

    save_pipeline(
        model=model,
        scaler=artifacts["scaler"],
        encoder_product=artifacts["encoder_product"],
        encoder_vendor=artifacts["encoder_vendor"],
        metadata=metadata,
        output_dir=output_dir,
    )

    print(f"Pipeline saved to {output_dir}")
    print(f"Config: {CONFIG}")
    print(f"Validation precision: {metrics['validation_precision']:.4f}")
    return output_dir


# Inference


def load_pipeline(
    pipeline_dir: str | Path | None = None,
) -> tuple[LogisticRegression, StandardScaler, TargetEncoder, TargetEncoder, dict]:
    """Load model, scaler, encoders and metadata from a pipeline directory.
    Default: models/purchase_model_mvp_20260307 (relative to this script).
    """
    if pipeline_dir is None:
        pipeline_dir = DEFAULT_PIPELINE_DIR
    pipeline_dir = Path(pipeline_dir)
    with open(pipeline_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(pipeline_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(pipeline_dir / "encoder_product.pkl", "rb") as f:
        encoder_product = pickle.load(f)
    with open(pipeline_dir / "encoder_vendor.pkl", "rb") as f:
        encoder_vendor = pickle.load(f)
    with open(pipeline_dir / "metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, encoder_product, encoder_vendor, metadata


def predict_proba(
    X_raw: pd.DataFrame, pipeline_dir: str | Path | None = None
) -> np.ndarray:
    """
    Load pipeline and run inference on raw data.
    X_raw must contain all columns required for feature_cols (including product_type and vendor for encoding).
    Returns probability of positive class (purchase).
    pipeline_dir defaults to models/purchase_model_mvp_20260307.
    """
    model, scaler, encoder_product, encoder_vendor, metadata = load_pipeline(
        pipeline_dir
    )
    feature_cols = metadata["feature_cols"]

    df = X_raw.copy()
    if "product_type_enc" not in df.columns and "product_type" in df.columns:
        df["product_type_enc"] = encoder_product.transform(df[["product_type"]])
    if "vendor_enc" not in df.columns and "vendor" in df.columns:
        df["vendor_enc"] = encoder_vendor.transform(df[["vendor"]])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for inference: {missing}")

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


if __name__ == "__main__":
    main()
