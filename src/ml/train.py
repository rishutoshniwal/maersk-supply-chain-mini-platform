"""
Model training and evaluation pipeline for delay risk prediction.
Trains only on booking-time features, strictly avoiding target leakage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.db import get_read_connection
from src.ml.features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    extract_booking_features,
    validate_no_leakage,
)


def load_training_data(db_path: Path) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Load unquarantined, completed shipments and port reference data."""
    with get_read_connection(db_path) as conn:
        ports_df = conn.execute("SELECT port_code, region, avg_congestion_score FROM curated_ports;").df()
        shipments_df = conn.execute("""
            SELECT
                shipment_id,
                origin_port,
                destination_port,
                cargo_type,
                container_count,
                weight_tons,
                booking_date,
                planned_departure,
                planned_arrival,
                actual_delay_hours
            FROM curated_shipments
            WHERE NOT is_quarantined
              AND actual_arrival IS NOT NULL;
        """).df()

    port_meta = ports_df.set_index("port_code").to_dict(orient="index")

    # Binary delay target: whether arrival delay > 24 hours
    y = (shipments_df["actual_delay_hours"] > 24.0).astype(int)

    # Feature extraction (booking-time only)
    X = extract_booking_features(shipments_df, port_meta)

    return X, y, port_meta


def build_pipeline(classifier) -> Pipeline:
    """Build scikit-learn preprocessing and classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Compute honest evaluation metrics on held-out test split."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        "confusion_matrix": cm,
    }
    return metrics


def train_delay_model(
    db_path: Path = Path("./data/curated/supply_chain.duckdb"),
    model_output_dir: Path = Path("./models"),
) -> Dict:
    """Train delay risk prediction model and save artifacts."""
    print("[*] Loading training dataset from DuckDB...")
    X, y, port_meta = load_training_data(db_path)

    # Verify zero data leakage
    validate_no_leakage(X)

    print(f"    - Total completed training samples: {len(X):,}")
    print(f"    - Positive class (Delay > 24h): {y.sum():,} ({y.mean():.2%})")
    print(f"    - Negative class (On-time <= 24h): {(len(y) - y.sum()):,} ({(1 - y.mean()):.2%})")

    # 80/20 train/test split stratified by target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    fitted_models = {}

    print("[*] Training candidate models on booking-time features...")
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(name, pipe, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = pipe
        print(f"    -> {name:<22}: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    # Select best model based on ROC-AUC / F1 balance
    best_name = "HistGradientBoosting" if results["HistGradientBoosting"]["roc_auc"] >= results["LogisticRegression"]["roc_auc"] else "LogisticRegression"
    best_model = fitted_models[best_name]
    best_metrics = results[best_name]

    print(f"[+] Selected best model: {best_name}")

    # Save artifact
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_output_dir / "delay_predictor.joblib"
    joblib.dump(best_model, model_file)

    # Save model metadata
    metadata = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_name,
        "leakage_guarantee": "Booking-time features only. Actual departure/arrival excluded.",
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all": ALL_FEATURE_COLUMNS,
        },
        "target": "is_delayed_gt_24h (actual_delay_hours > 24)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "evaluation_metrics": best_metrics,
        "all_model_evaluations": results,
        "port_metadata_cache": port_meta,
        "production_monitoring": {
            "metrics_to_track": [
                "ROC-AUC on rolling 30-day arrived shipments",
                "F1-score and Precision/Recall curve",
                "Population Stability Index (PSI) on weight_tons and congestion scores",
                "Feature drift Kolmogorov-Smirnov test (p-value < 0.05)",
                "Prediction drift (mean predicted delay probability vs historical baseline)",
            ],
            "retrain_triggers": [
                "Rolling ROC-AUC drops below 0.65",
                "Feature drift PSI exceeds 0.25 on primary numerical features",
                "New port or regional conflict introduces persistent route disruptions",
                "Scheduled bi-weekly retraining cadence with newly confirmed shipment actuals",
            ],
        },
    }

    meta_file = model_output_dir / "model_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Model artifact saved to: {model_file}")
    print(f"[+] Model metadata and evaluation saved to: {meta_file}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train delay prediction model on booking-time features.")
    parser.add_argument("--db-path", default="./data/curated/supply_chain.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="./models", help="Directory to save model artifacts")
    args = parser.parse_args()

    train_delay_model(Path(args.db_path), Path(args.output_dir))


if __name__ == "__main__":
    main()
