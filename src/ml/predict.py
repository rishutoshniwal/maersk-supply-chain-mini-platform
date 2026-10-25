"""
Inference service for delay risk prediction.
Loads committed model artifact into memory for sub-millisecond predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.ml.features import extract_booking_features


class DelayPredictor:
    """Thread-safe, high-performance model predictor."""

    def __init__(
        self,
        model_path: Path = Path("./models/delay_predictor.joblib"),
        metadata_path: Path = Path("./models/model_metadata.json"),
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.port_meta: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load model pipeline and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
        self.model = joblib.load(self.model_path)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.port_meta = self.metadata.get("port_metadata_cache", {})

    def predict(self, booking_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prediction from a booking attribute dictionary.
        Returns probability, delay flag, and risk level.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        # Ensure single row dataframe
        df = pd.DataFrame([booking_payload])
        feature_matrix = extract_booking_features(df, self.port_meta)

        prob = float(self.model.predict_proba(feature_matrix)[0, 1])
        predicted_delayed = bool(prob >= 0.50)

        # Risk level categorization
        if prob < 0.35:
            risk_level = "LOW"
        elif prob < 0.60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "delay_probability": round(prob, 4),
            "predicted_delayed": predicted_delayed,
            "delay_threshold_hours": 24.0,
            "risk_level": risk_level,
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "model_type": self.metadata.get("selected_model", "HistGradientBoosting"),
        }


# Singleton instance for fast API startup
_predictor_instance: Optional[DelayPredictor] = None


def get_predictor() -> DelayPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DelayPredictor()
    return _predictor_instance
